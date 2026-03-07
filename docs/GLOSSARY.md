# Glossary

This glossary defines the main topology-optimization, FEM, and project-specific terms used in TopoptComec.

## Optimization Terms

### SIMP

Solid Isotropic Material with Penalization. A standard topology optimization method where element stiffness is interpolated from a density value and penalized to discourage intermediate densities.

In this project, SIMP is the main optimization formulation used by `app/core/optimizers.py`.

### OC

Optimality Criteria. An iterative update rule used to change element densities while respecting a volume constraint. In this repository, `_oc(...)` applies move limits and a bisection search on the Lagrange multiplier.

### Density Field

The array of per-element design variables. Values are usually between `0` and `1`:

- near `0`: void / removed material
- near `1`: solid material
- in between: intermediate material, usually undesirable in the final design

### `x`

The raw design variable array before all physical projections and region overrides are applied.

### `xPhys`

The physical density field used by FEM solves and export. It usually means the current design after filtering and region application. In multi-material mode it may be a 2D array shaped like `(n_materials, n_elements)`.

### Volume Fraction (`volfrac`)

The target fraction of the design domain that may remain filled with material. Lower values produce lighter structures but can make the optimization problem harder.

### Penalization (`penal`)

The exponent applied in the SIMP stiffness interpolation. Higher values penalize gray regions more aggressively.

### Filter Radius (`filter_radius_min`)

The neighborhood size used for density or sensitivity filtering. It reduces numerical artifacts such as checkerboarding and mesh dependency.

### Max Change (`max_change`)

The maximum density change allowed per optimization iteration by the OC update.

### Eta (`eta`)

The damping exponent in the OC update rule. It affects how aggressively the design changes between iterations.

## FEM Terms

### FEM

Finite Element Method. The numerical method used here to approximate elastic deformation of the structure.

### Element

A discrete cell of the design grid. In this project the topology optimization variables are stored per element, not per node.

### Node

A mesh point where displacements are defined and where loads or supports may be applied.

### DOF

Degree of Freedom. A single displacement component associated with a node.

- 2D: each node has `X` and `Y` DOFs
- 3D: each node has `X`, `Y`, and `Z` DOFs

### Stiffness Matrix (`K`)

The sparse linear system assembled from the element stiffness matrix and the current material distribution. Solving `K u = f` yields nodal displacements.

### `KE`

The element stiffness matrix used as the local building block for the global stiffness matrix.

### Boundary Conditions

The constraints and loads applied to the model:

- supports fix some DOFs
- forces load some DOFs

### Plane 2D vs 3D

The code switches behavior based on `nelz`:

- `nelz == 0`: 2D mode
- `nelz > 0`: 3D mode

This convention appears across the UI, FEM layer, plotting, and export code.

## Mechanism Terms

### Rigid Mechanism Case

In this repository, a case with input forces and supports but no output force. The objective behaves like compliance minimization: keep the structure stiff.

### Compliant Mechanism Case

A case with both input and output forces. The objective is based on transfer behavior, not only stiffness. The solver tries to create elastic motion transmission through deformation.

### Input Force

An actuating force defined by:

- position
- direction
- magnitude

Input forces are stored under `Forces["fi*"]`.

### Output Force

A target response location and direction used to evaluate compliant mechanism behavior. Output forces are stored under `Forces["fo*"]`.

### Efficiency

A heuristic post-analysis metric computed in `app/core/analyzers.py`. It is based on displacement behavior and is used as a quick quality check, not as a rigorous engineering certification.

## Project-Specific Terms

### Region

A geometric override applied to the density field before or during optimization.

Typical uses:

- force a zone to stay solid
- force a zone to stay void

### `rshape`, `rstate`, `rradius`

Region fields in the parameter schema:

- `rshape`: shape selector, such as circle or square
- `rstate`: whether the region is forced to `Void` or `Solid`
- `rradius`: size control

### Support Radius (`sr`)

An optional radius around a support point. When it is non-zero, all nodes inside that neighborhood are constrained according to the selected support dimensions.

### Thresholding

Turning a gray density field into a binary one, usually with a cutoff of `0.5`. The CLI exposes this through `-t`.

### Checkerboarding

A non-physical alternating solid/void pattern caused by discretization artifacts. The project includes a heuristic detector for this in post-analysis.

### Watertight

Used here as a practical connectedness check on the thresholded structure. The analyzer verifies whether the binarized design forms a single connected component.

### Warping / Displacement Visualization

Post-processing that distorts the displayed geometry using displacement data so the user can inspect how a mechanism moves.

### Preset

A named parameter set stored in `presets.json`. Presets are the main way to reproduce designs across GUI and CLI runs.
