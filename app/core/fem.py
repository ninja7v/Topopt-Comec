# app/core/fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Finite Element Method (FEM) class for topology optimization.

from typing import Dict, Tuple
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, cg, LinearOperator


class FEM:
    def __init__(self, Dimensions: Dict, Materials: Dict, Optimizer: Dict):
        # Geometry and Grid Setup
        self.nelxyz = Dimensions.get("nelxyz", [1, 1, 1])
        self.nelx, self.nely, self.nelz = self.nelxyz
        self.is_3d = self.nelz > 0
        self.nel = self.nelx * self.nely * (self.nelz if self.is_3d else 1)
        self.elemndof = 3 if self.is_3d else 2
        self.dim_mul = self.elemndof
        self.ndof = (
            self.dim_mul
            * (self.nelx + 1)
            * (self.nely + 1)
            * ((self.nelz + 1) if self.is_3d else 1)
        )

        # Materials and Optimization Params
        self.E_min, self.E_max = 1e-9, Materials.get("E", [1.0])[0]
        self.nu = Materials.get("nu", [0.3])[0]
        self.penal = Optimizer.get("penal", 3.0)
        self.solver_type = Optimizer.get("solver", "default")
        self.filter_type = Optimizer.get("filter_type", 0)
        self.filter_radius = Optimizer.get("filter_radius_min", 0.0)

        # Pre-compute Constant Matrices (KE, DOF Maps, Filter)
        self.KE = self._get_lk_stiffness()
        self.edofMat, self.iK, self.jK = self._build_dof_map()
        self.H, self.Hs = self._build_filter()

        # State placeholders for BCs
        self.fixed_dofs = np.array([], dtype=int)
        self.free_dofs = np.array([], dtype=int)
        self.fi_indices = []
        self.fo_indices = []
        self.forces_i = None
        self.forces_o = None

    def setup_boundary_conditions(self, Forces: Dict, Supports: Dict = None):
        """Parses Forces and Supports dicts to create force vectors and fixed DOFs."""
        # Forces
        self.forces_i, self.fi_indices, di = self._parse_forces(
            Forces, "fi", "fix", "fiy", "fiz", "fidir", "finorm"
        )
        self.forces_o, self.fo_indices, do = self._parse_forces(
            Forces, "fo", "fox", "foy", "foz", "fodir", "fonorm"
        )

        self.di_indices = di  # For artificial stiffness addition
        self.do_indices = do  # For artificial stiffness addition
        self.finorm = Forces.get("finorm", [])
        self.fonorm = Forces.get("fonorm", [])

        # Supports
        fixed = []
        if Supports:
            sx, sy, sz = (
                Supports.get("sx", []),
                Supports.get("sy", []),
                Supports.get("sz", []),
            )
            sr = Supports.get("sr", [0] * len(sx))
            sdim = Supports.get("sdim", [])
            active_sup = [i for i, val in enumerate(sdim) if val != "-"]

            for i in active_sup:
                center_node_idx = self._get_node_idx(
                    sx[i], sy[i], sz[i] if self.is_3d else 0
                )
                nodes_to_fix = [center_node_idx]

                # If radius > 0, find all nodes within radius
                if i < len(sr) and sr[i] > 0:
                    radius = sr[i]
                    # Determine range to search
                    x_range = range(
                        max(0, int(sx[i] - radius)),
                        min(self.nelx + 1, int(sx[i] + radius + 1)),
                    )
                    y_range = range(
                        max(0, int(sy[i] - radius)),
                        min(self.nely + 1, int(sy[i] + radius + 1)),
                    )
                    z_range = (
                        range(
                            max(0, int(sz[i] - radius)),
                            min(self.nelz + 1, int(sz[i] + radius + 1)),
                        )
                        if self.is_3d
                        else range(1)
                    )

                    for z in z_range:
                        for x in x_range:
                            for y in y_range:
                                dist_sq = (
                                    (x - sx[i]) ** 2
                                    + (y - sy[i]) ** 2
                                    + ((z - sz[i]) ** 2 if self.is_3d else 0)
                                )
                                if dist_sq <= radius**2:
                                    n_idx = self._get_node_idx(
                                        x, y, z if self.is_3d else 0
                                    )
                                    nodes_to_fix.append(n_idx)

                for node_idx in nodes_to_fix:
                    node_dof = self.dim_mul * node_idx
                    if "X" in sdim[i]:
                        fixed.append(node_dof)
                    if "Y" in sdim[i]:
                        fixed.append(node_dof + 1)
                    if self.is_3d and "Z" in sdim[i]:
                        fixed.append(node_dof + 2)

        self.fixed_dofs = np.unique(fixed)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)

    def apply_regions(self, x: np.ndarray, Regions: Dict) -> np.ndarray:
        """Applies geometric constraints (Regions) to the density field."""
        xPhys = x.copy()
        rshape = Regions.get("rshape", [])
        if not rshape:
            return xPhys

        rx, ry, rz = Regions.get("rx", []), Regions.get("ry", []), Regions.get("rz", [])
        rradius, rstate = Regions.get("rradius", []), Regions.get("rstate", [])

        active_regions = [i for i, s in enumerate(rshape) if s != "-"]

        # This loop logic is specific to the grid geometry
        # Simplified for brevity, assumes standard implementation from original code
        for i in active_regions:
            val = 1e-6 if rstate[i] == "Void" else 1.0
            r = rradius[i]
            z_range = (
                range(max(0, int(rz[i] - r)), min(self.nelz, int(rz[i] + r)))
                if self.is_3d
                else range(1)
            )

            for ez in z_range:
                for ex in range(max(0, int(rx[i] - r)), min(self.nelx, int(rx[i] + r))):
                    for ey in range(
                        max(0, int(ry[i] - r)), min(self.nely, int(ry[i] + r))
                    ):
                        # Geometric check
                        if rshape[i] == "◯":
                            dist_sq = (
                                (ex - rx[i]) ** 2
                                + (ey - ry[i]) ** 2
                                + ((ez - rz[i]) ** 2 if self.is_3d else 0)
                            )
                            if dist_sq > r**2:
                                continue

                        idx = (
                            (ez * self.nelx * self.nely if self.is_3d else 0)
                            + ex * self.nely
                            + ey
                        )
                        xPhys[idx] = val
        return xPhys

    def solve(self, xPhys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Assembles K and solves for Input and Output forces."""
        # Assembly
        sK = (self.KE.flatten()[np.newaxis]).T * (
            self.E_min + xPhys**self.penal * (self.E_max - self.E_min)
        )
        K = coo_matrix(
            (sK.flatten(order="F"), (self.iK, self.jK)), shape=(self.ndof, self.ndof)
        ).tocsc()
        # Add artificial stiffness at force locations
        self._add_artificial_springs(K, self.di_indices, self.fi_indices, self.finorm)
        self._add_artificial_springs(K, self.do_indices, self.fo_indices, self.fonorm)

        # Solving
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        ui = np.zeros((self.ndof, len(self.fi_indices)))
        uo = np.zeros((self.ndof, len(self.fo_indices)))
        self._solve_linear_system(K_free, self.forces_i, self.fi_indices, ui)
        self._solve_linear_system(K_free, self.forces_o, self.fo_indices, uo)

        return ui, uo

    def solve_with_E_eff(self, E_eff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assembles K using pre-computed effective element stiffness and solves."""
        sK = (self.KE.flatten()[np.newaxis]).T * E_eff
        K = coo_matrix(
            (sK.flatten(order="F"), (self.iK, self.jK)), shape=(self.ndof, self.ndof)
        ).tocsc()
        self._add_artificial_springs(K, self.di_indices, self.fi_indices, self.finorm)
        self._add_artificial_springs(K, self.do_indices, self.fo_indices, self.fonorm)

        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        ui = np.zeros((self.ndof, len(self.fi_indices)))
        uo = np.zeros((self.ndof, len(self.fo_indices)))
        self._solve_linear_system(K_free, self.forces_i, self.fi_indices, ui)
        self._solve_linear_system(K_free, self.forces_o, self.fo_indices, uo)

        return ui, uo

    def compute_ce(self, ui: np.ndarray, uo: np.ndarray) -> np.ndarray:
        """Compute element compliance ce = u^T KE u for all elements.

        For rigid mechanisms (no output forces): ce = sum_i u_i^T KE u_i
        For compliant mechanisms: ce = sum_i sum_o u_in^T KE u_out
        """
        nb_out = len(self.fo_indices)
        ce_total = np.zeros(self.nel)

        if nb_out == 0:  # Rigid Mechanism (Minimize Compliance)
            for i_in in self.fi_indices:
                Ue = ui[self.edofMat, i_in]
                if self.is_3d:
                    ce_total += np.sum(np.dot(Ue, self.KE) * Ue, axis=1)
                else:
                    ce_total += np.einsum("ij,jk,ik->i", Ue, self.KE, Ue)
        else:  # Compliant Mechanism
            if self.is_3d:
                for el in range(self.nel):
                    for i_in in self.fi_indices:
                        Ue_in = ui[self.edofMat[el, :], [i_in]]
                        for i_out in self.fo_indices:
                            Ue_out = uo[self.edofMat[el, :], [i_out]]
                            ce_total[el] += (Ue_in.T @ self.KE @ Ue_out).item()
            else:
                for i_in in self.fi_indices:
                    Ue_in = ui[self.edofMat, i_in]
                    for i_out in self.fo_indices:
                        Ue_out = uo[self.edofMat, i_out]
                        ce_total += np.einsum("ij,jk,ik->i", Ue_in, self.KE, Ue_out)

        return ce_total

    def compute_sensitivities(
        self, xPhys: np.ndarray, ui: np.ndarray, uo: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculates Objective value, Sensitivity (dc), and Volume Sensitivity (dv)."""
        nb_out = len(self.fo_indices)
        ce_total = self.compute_ce(ui, uo)

        # 1. Compliance / Objective Calculation
        obj_val = 0.0
        if nb_out == 0:  # Rigid Mechanism (Minimize Compliance)
            obj_val = (
                (self.E_min + xPhys**self.penal * (self.E_max - self.E_min)) * ce_total
            ).sum()
            dc = -self.penal * (xPhys ** (self.penal - 1)) * ce_total
        else:  # Compliant Mechanism
            dc = self.penal * (xPhys ** (self.penal - 1)) * ce_total
            # Sum of absolute output displacements
            for idx, dof_indices in enumerate(self.do_indices):
                obj_val += sum(abs(uo[dof, idx]) for dof in [dof_indices]) / nb_out

        # 2. Volume Sensitivity (dv)
        dv = np.ones(self.nel)

        # 3. Filtering
        return obj_val, self._apply_filter(xPhys, dc, dv)

    # --- Internal Helper Methods ---

    def _get_node_idx(self, x, y, z):
        return (
            (z * (self.nelx + 1) * (self.nely + 1) if self.is_3d else 0)
            + x * (self.nely + 1)
            + y
        )

    def _parse_forces(
        self,
        Forces: Dict,
        prefix: str,
        kx: str,
        ky: str,
        kz: str,
        kdir: str,
        knorm: str,
    ):
        fx, fy = Forces.get(kx, []), Forces.get(ky, [])
        fz = Forces.get(kz, []) if self.is_3d else []
        fdir = Forces.get(kdir, [])

        active_indices = [i for i, val in enumerate(fdir) if val != "-"]
        f_vec = np.zeros((self.ndof, len(active_indices)))
        dof_indices = []

        for mat_idx, i in enumerate(active_indices):
            node = self._get_node_idx(fx[i], fy[i], fz[i] if self.is_3d else 0)
            val = 4 / 100.0  # Artificial stiffness value for force application points
            dof = -1
            if "X" in fdir[i]:
                dof = self.dim_mul * node
                if "←" in fdir[i]:
                    val = -val
            elif "Y" in fdir[i]:
                dof = self.dim_mul * node + 1
                if "↑" in fdir[i]:
                    val = -val
            elif self.is_3d and "Z" in fdir[i]:
                dof = self.dim_mul * node + 2
                if ">" in fdir[i]:
                    val = -val

            f_vec[dof, mat_idx] = val
            dof_indices.append(dof)

        return f_vec, active_indices, dof_indices

    def _add_artificial_springs(self, K, dofs, active_indices, norms):
        for i, dof in enumerate(dofs):
            original_idx = active_indices[i]
            if original_idx < len(norms) and norms[original_idx] > 0:
                K[dof, dof] += norms[original_idx]

    def _solve_linear_system(self, K_free, F, active_indices, U_full):
        if not active_indices:
            return

        use_direct = self.solver_type == "Direct" or (
            self.solver_type == "Auto" and K_free.shape[0] < 10000
        )

        if use_direct:
            for i in range(len(active_indices)):
                if np.any(F[self.free_dofs, i]):
                    U_full[self.free_dofs, i] = spsolve(K_free, F[self.free_dofs, i])
        else:
            # Iterative Solver (CG with Jacobi Preconditioner)
            D_inv = 1.0 / K_free.diagonal()
            M = LinearOperator(K_free.shape, lambda x: D_inv * x)
            for i in range(len(active_indices)):
                if np.any(F[self.free_dofs, i]):
                    u_sol, info = cg(
                        K_free,
                        F[self.free_dofs, i],
                        M=M,
                        rtol=1e-6,
                        maxiter=K_free.shape[0],
                    )
                    if info != 0 and self.solver_type == "Auto":
                        # Fallback
                        try:
                            U_full[self.free_dofs, i] = spsolve(
                                K_free, F[self.free_dofs, i]
                            )
                        except Exception as e:
                            print(
                                f"Direct solver failed: {e}. Using partial CG solution."
                            )
                            U_full[self.free_dofs, i] = u_sol
                    else:
                        U_full[self.free_dofs, i] = u_sol

    def _apply_filter(self, x, dc, dv):
        if self.filter_type == "Sensitivity":
            # H * (x * dc) / Hs / max(x, 0.001)
            dc = np.asarray((self.H @ (x * dc)) / self.Hs.flatten()) / np.maximum(
                0.001, x
            )
        elif self.filter_type == "Density":
            dc = np.asarray(self.H * (dc[np.newaxis].T / self.Hs))[:, 0]
            dv = np.asarray(self.H * (dv[np.newaxis].T / self.Hs))[:, 0]
        return dc, dv

    def update_xPhys(self, x) -> np.ndarray:
        """Calculates physical density based on design variable and filter."""
        if self.filter_type == "Density":
            return (self.H @ x).ravel() / np.asarray(self.Hs).ravel()
        return x

    def compute_element_stiffness(self, rho: np.ndarray, E_list: list) -> np.ndarray:
        """Compute effective per-element stiffness for multi-material SIMP.

        Args:
            rho: Material densities, shape (n_mat, nel).
            E_list: Young's modulus for each material.

        Returns:
            Effective element stiffness array of shape (nel,).
        """
        E_eff = np.full(self.nel, self.E_min)
        for i, E_i in enumerate(E_list):
            E_eff += rho[i] ** self.penal * E_i
        return E_eff

    def _get_lk_stiffness(self) -> np.ndarray:
        """Get element stiffness matrix."""
        E, nu = 1.0, self.nu  # Normalized E for KE
        if self.is_3d:
            A = np.array(
                [
                    [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                    [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
                ]
            )
            k = 1 / 72 * (A.T @ np.array([1, nu]))
            K_blocks = self._build_3d_blocks(k)
            return E / ((nu + 1) * (1 - 2 * nu)) * K_blocks
        else:
            k = np.array(
                [
                    1 / 2 - nu / 6,
                    1 / 8 + nu / 8,
                    -1 / 4 - nu / 12,
                    -1 / 8 + 3 * nu / 8,
                    -1 / 4 + nu / 12,
                    -1 / 8 - nu / 8,
                    nu / 6,
                    1 / 8 - 3 * nu / 8,
                ]
            )
            return (
                E
                / (1 - nu**2)
                * np.array(
                    [
                        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                    ]
                )
            )

    def _build_3d_blocks(self, k) -> np.ndarray:
        K1 = np.array(
            [
                [k[0], k[1], k[1], k[2], k[4], k[4]],
                [k[1], k[0], k[1], k[3], k[5], k[6]],
                [k[1], k[1], k[0], k[3], k[6], k[5]],
                [k[2], k[3], k[3], k[0], k[7], k[7]],
                [k[4], k[5], k[6], k[7], k[0], k[1]],
                [k[4], k[6], k[5], k[7], k[1], k[0]],
            ]
        )
        K2 = np.array(
            [
                [k[8], k[7], k[11], k[5], k[3], k[6]],
                [k[7], k[8], k[11], k[4], k[2], k[4]],
                [k[9], k[9], k[12], k[6], k[3], k[5]],
                [k[5], k[4], k[10], k[8], k[1], k[9]],
                [k[3], k[2], k[4], k[1], k[8], k[11]],
                [k[10], k[3], k[5], k[11], k[9], k[12]],
            ]
        )
        K3 = np.array(
            [
                [k[5], k[6], k[3], k[8], k[11], k[7]],
                [k[6], k[5], k[3], k[9], k[12], k[9]],
                [k[4], k[4], k[2], k[7], k[11], k[8]],
                [k[8], k[9], k[1], k[5], k[10], k[4]],
                [k[11], k[12], k[9], k[10], k[5], k[3]],
                [k[1], k[11], k[8], k[3], k[4], k[2]],
            ]
        )
        K4 = np.array(
            [
                [k[13], k[10], k[10], k[12], k[9], k[9]],
                [k[10], k[13], k[10], k[11], k[8], k[7]],
                [k[10], k[10], k[13], k[11], k[7], k[8]],
                [k[12], k[11], k[11], k[13], k[6], k[6]],
                [k[9], k[8], k[7], k[6], k[13], k[10]],
                [k[9], k[7], k[8], k[6], k[10], k[13]],
            ]
        )
        K5 = np.array(
            [
                [k[0], k[1], k[7], k[2], k[4], k[3]],
                [k[1], k[0], k[7], k[3], k[5], k[10]],
                [k[7], k[7], k[0], k[4], k[10], k[5]],
                [k[2], k[3], k[4], k[0], k[7], k[1]],
                [k[4], k[5], k[10], k[7], k[0], k[7]],
                [k[3], k[10], k[5], k[1], k[7], k[0]],
            ]
        )
        K6 = np.array(
            [
                [k[13], k[10], k[6], k[12], k[9], k[11]],
                [k[10], k[13], k[6], k[11], k[8], k[1]],
                [k[6], k[6], k[13], k[9], k[1], k[8]],
                [k[12], k[11], k[9], k[13], k[6], k[10]],
                [k[9], k[8], k[1], k[6], k[13], k[6]],
                [k[11], k[1], k[8], k[10], k[6], k[13]],
            ]
        )
        return np.block(
            [
                [K1, K2, K3, K4],
                [K2.T, K5, K6, K3.T],
                [K3.T, K6, K5.T, K2.T],
                [K4, K3, K2, K1.T],
            ]
        )

    def _build_dof_map(self):
        size = 8 * (self.elemndof if self.is_3d else 1)
        edofMat = np.zeros((self.nel, size), dtype=int)
        # Todo: vectorize this loop for performance
        for ex in range(self.nelx):
            for ey in range(self.nely):
                for ez in range(self.nelz if self.is_3d else 1):
                    n1 = (
                        (ez * (self.nelx + 1) * (self.nely + 1) if self.is_3d else 0)
                        + ex * (self.nely + 1)
                        + ey
                    )
                    nodes = [
                        n1 + 1,
                        n1 + self.nely + 2,
                        n1 + self.nely + 1,
                        n1,
                    ]  # Bottom face
                    if self.is_3d:
                        off = (self.nelx + 1) * (self.nely + 1)
                        nodes += [n + off for n in nodes]

                    dofs = []
                    for n in nodes:
                        dofs.extend([n * self.dim_mul + d for d in range(self.dim_mul)])

                    el = (
                        (ez * self.nelx * self.nely if self.is_3d else 0)
                        + ex * self.nely
                        + ey
                    )
                    edofMat[el, :] = dofs

        iK = np.kron(edofMat, np.ones((size, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, size))).flatten()
        return edofMat, iK, jK

    def _build_filter(self):
        r = np.ceil(self.filter_radius)
        iH, jH, sH = [], [], []
        # Todo: vectorize this loop for performance
        for ez in range(self.nelz if self.is_3d else 1):
            for ex in range(self.nelx):
                for ey in range(self.nely):
                    el1 = (
                        (ez * self.nelx * self.nely if self.is_3d else 0)
                        + ex * self.nely
                        + ey
                    )

                    k_min, k_max = (
                        (max(ez - r + 1, 0), min(ez + r, self.nelz))
                        if self.is_3d
                        else (0, 1)
                    )
                    i_min, i_max = max(ex - r + 1, 0), min(ex + r, self.nelx)
                    j_min, j_max = max(ey - r + 1, 0), min(ey + r, self.nely)

                    for k in range(int(k_min), int(k_max)):
                        for i in range(int(i_min), int(i_max)):
                            for j in range(int(j_min), int(j_max)):
                                el2 = (
                                    (k * self.nelx * self.nely if self.is_3d else 0)
                                    + i * self.nely
                                    + j
                                )
                                dist = np.sqrt(
                                    (ex - i) ** 2
                                    + (ey - j) ** 2
                                    + ((ez - k) ** 2 if self.is_3d else 0)
                                )
                                val = self.filter_radius - dist
                                if val > 0:
                                    iH.append(el1)
                                    jH.append(el2)
                                    sH.append(val)

        H = coo_matrix((sH, (iH, jH)), shape=(self.nel, self.nel)).tocsc()
        return H, H.sum(1)
