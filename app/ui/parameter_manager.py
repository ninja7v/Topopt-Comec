# app/ui/parameter_manager.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Manage parameters (json file to UI, comparison, ...).

import copy
from PySide6.QtWidgets import QMessageBox


class ParameterManagerMixin:
    """Mixin for MainWindow to handle parameter gathering, validation, and equivalency checks."""

    def gather_parameters(self):
        """Collects all parameters from the UI controls into a nested dictionary."""
        params = {}

        # --- Dimensions ---
        nelx, nely, nelz = (
            self.dim_widget.nx.value(),
            self.dim_widget.ny.value(),
            self.dim_widget.nz.value(),
        )
        params["Dimensions"] = {
            "nelxyz": [nelx, nely, nelz],
            "volfrac": self.dim_widget.volfrac.value(),
        }

        # --- Regions (optional) ---
        Regions = {
            "rshape": [],
            "rstate": [],
            "rradius": [],
            "rx": [],
            "ry": [],
            "rz": [],
        }
        for rw in self.regions_widget.inputs:
            Regions["rshape"].append(rw["rshape"].currentText())
            Regions["rstate"].append(rw["rstate"].currentText())
            Regions["rradius"].append(rw["rradius"].value())
            Regions["rx"].append(rw["rx"].value())
            Regions["ry"].append(rw["ry"].value())
            Regions["rz"].append(rw["rz"].value())
        if Regions["rshape"]:  # only add if there is at least one region
            params["Regions"] = Regions

        # --- Forces ---
        Forces = {
            "fix": [],
            "fiy": [],
            "fiz": [],
            "fidir": [],
            "finorm": [],
            "fox": [],
            "foy": [],
            "foz": [],
            "fodir": [],
            "fonorm": [],
        }
        for fw in self.forces_widget.inputs:
            if "fix" in fw:  # Input force
                Forces["fix"].append(fw["fix"].value())
                Forces["fiy"].append(fw["fiy"].value())
                Forces["fiz"].append(fw["fiz"].value())
                Forces["fidir"].append(fw["fidir"].currentText())
                Forces["finorm"].append(fw["finorm"].value())
            elif "fox" in fw:  # Output force
                Forces["fox"].append(fw["fox"].value())
                Forces["foy"].append(fw["foy"].value())
                Forces["foz"].append(fw["foz"].value())
                Forces["fodir"].append(fw["fodir"].currentText())
                Forces["fonorm"].append(fw["fonorm"].value())
        params["Forces"] = Forces

        # --- Supports (optional) ---
        Supports = {"sx": [], "sy": [], "sz": [], "sdim": [], "sr": []}
        for sw in self.supports_widget.inputs:
            Supports["sx"].append(sw["sx"].value())
            Supports["sy"].append(sw["sy"].value())
            Supports["sz"].append(sw["sz"].value())
            Supports["sdim"].append(sw["sdim"].currentText())
            Supports["sr"].append(sw["sr"].value())
        if Supports["sx"]:
            params["Supports"] = Supports

        # --- Materials ---
        Materials = {"E": [], "nu": [], "percent": [], "color": []}
        for mat in self.materials_widget.inputs:
            Materials["E"].append(mat["E"].value())
            Materials["nu"].append(mat["nu"].value())
            Materials["percent"].append(mat["percent"].value())
            Materials["color"].append(mat["color"].text())
        Materials["init_type"] = self.materials_widget.mat_init_type.currentIndex()
        params["Materials"] = Materials

        # --- Optimizer ---
        Optimizer = {
            "filter_type": (
                "Sensitivity"
                if self.optimizer_widget.opt_ft.currentIndex() == 0
                else (
                    "Density"
                    if self.optimizer_widget.opt_ft.currentIndex() == 1
                    else "None"
                )
            ),
            "filter_radius_min": self.optimizer_widget.opt_fr.value(),
            "penal": self.optimizer_widget.opt_p.value(),
            "eta": self.optimizer_widget.opt_eta.value(),
            "max_change": self.optimizer_widget.opt_max_change.value(),
            "n_it": self.optimizer_widget.opt_n_it.value(),
            "solver": self.optimizer_widget.opt_solver.currentText(),
        }
        params["Optimizer"] = Optimizer

        # --- Displacement (optional) ---
        Displacement = {
            "disp_factor": self.displacement_widget.mov_disp.value(),
            "disp_iterations": self.displacement_widget.mov_iter.value(),
        }
        params["Displacement"] = Displacement

        return params

    def on_parameter_changed(self):
        """React when a parameter is changed."""
        # Play the animation
        self.footer.start_create_button_effect()

        # First, check if a valid result from a previous run exists.
        if self.xPhys is not None:
            self.xPhys = None
            self.u = None
            self.is_displaying_deformation = False

            # Disable buttons that require a valid result
            self.footer.binarize_button.setEnabled(False)
            self.footer.save_button.setEnabled(False)
            self.analysis_widget.run_analysis_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.sections["Displacement"].visibility_button.setEnabled(False)

            # Reset the analysis
            self.analysis_widget.checkerboard_result.setText("-")
            self.analysis_widget.watertight_result.setText("-")
            self.analysis_widget.threshold_result.setText("-")
            self.analysis_widget.efficiency_result.setText("-")

            # Inform the user what happened
            self.status_bar.showMessage(
                "Parameters changed. Please run 'Create' for a new result.", 3000
            )

        self.last_params = self.gather_parameters()
        self.replot()

        # Check if the current state matches the selected preset
        current_preset_name = self.preset.presets_combo.currentText()
        if current_preset_name in self.presets:
            if not self.are_parameters_equivalent(
                self.presets[current_preset_name], self.last_params
            ):
                # The parameters have changed, so deselect the preset
                self.preset.presets_combo.blockSignals(True)
                self.preset.presets_combo.setCurrentIndex(
                    0
                )  # Set to "Select a preset..."
                self.preset.presets_combo.blockSignals(False)
                self.preset.delete_preset_button.setEnabled(False)

    def are_parameters_equivalent(self, params1, params2):
        """Compares two parameter dictionaries, ignoring irrelevant data."""
        # Create deep copies to avoid modifying the original dictionaries
        p1 = copy.deepcopy(params1)
        p2 = copy.deepcopy(params2)

        self._normalize_params(p1)
        self._normalize_params(p2)

        return p1 == p2

    def _normalize_params(self, p):
        """Helper to normalize parameters for comparison."""
        pd = p["Dimensions"]
        if "nelxyz" in pd:
            is_2d = len(pd["nelxyz"]) < 3 or pd["nelxyz"][2] == 0.0
            if is_2d:
                pd["nelxyz"] = pd["nelxyz"][:2]

        self._normalize_regions(p, is_2d)
        self._normalize_supports(p, is_2d)
        self._normalize_forces(p, is_2d)
        self._normalize_materials(p)

        # Remove irrelevant keys
        p.pop("Displacement", None)
        p.pop("Optimizer", None)

    def _normalize_regions(self, p, is_2d):
        if "Regions" in p:
            pr = p["Regions"]
            if "rshape" in pr:
                zipped = zip(
                    pr.get("rshape", []),
                    pr.get("rstate", []),
                    pr.get("rradius", []),
                    pr.get("rx", []),
                    pr.get("ry", []),
                    pr.get("rz", []) if not is_2d else [0] * len(pr.get("rx", [])),
                )
                active = [r for r in zipped if r[0] != "-"]
                if active:
                    (
                        pr["rshape"],
                        pr["rstate"],
                        pr["rradius"],
                        pr["rx"],
                        pr["ry"],
                        pr["rz"],
                    ) = map(list, zip(*active))
                else:
                    for key in ["rshape", "rstate", "rradius", "rx", "ry", "rz"]:
                        pr.pop(key, None)  # pop them, not just empty them

                if is_2d and "rz" in pr:
                    pr.pop("rz")

    def _normalize_supports(self, p, is_2d):
        if "Supports" in p:
            ps = p["Supports"]
            if "sdim" in ps:
                zipped = zip(
                    ps.get("sx", []),
                    ps.get("sy", []),
                    ps.get("sz", []) if not is_2d else [0] * len(ps.get("sx", [])),
                    ps.get("sdim", []),
                    ps.get("sr", []),
                )
                active = [s for s in zipped if s[3] != "-"]
                if active:
                    (ps["sx"], ps["sy"], ps["sz"], ps["sdim"], ps["sr"]) = map(
                        list, zip(*active)
                    )
                else:
                    for key in ["sx", "sy", "sz", "sdim", "sr"]:
                        ps.pop(key, None)  # pop them, not just empty them
                if is_2d and "sz" in ps:
                    ps.pop("sz")

    def _normalize_forces(self, p, is_2d):
        pf = p["Forces"]
        for prefix in ["fi", "fo"]:
            dir_key = f"{prefix}dir"
            if dir_key in pf:
                keys = [
                    f"{prefix}x",
                    f"{prefix}y",
                    f"{prefix}z",
                    f"{prefix}dir",
                    f"{prefix}norm",
                ]
                vals = [pf.get(k, []) for k in keys]
                zipped = zip(*vals)
                active = []
                # Keep if direction is not "-"
                active = [item for item in zipped if item[3] != "-"]

                if active:
                    unzipped = list(zip(*active))
                    for i, k in enumerate(keys):
                        pf[k] = list(unzipped[i])
                else:
                    for k in keys:
                        pf[k] = []

                if is_2d and f"{prefix}z" in pf:
                    pf.pop(f"{prefix}z")

    def _normalize_materials(self, p):
        if "Materials" in p:
            pm = p["Materials"]
            if len(pm["E"]) == 1:
                # If only one material, ignore the percentage
                pm["percent"] = [100]
                if "color" not in pm or pm["color"] == [""]:
                    pm["color"] = ["#000000"]  # Default black color
                if "init_type" not in pm:
                    pm["init_type"] = 0

    def validate_parameters(self, params):
        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        if nelx <= 0 or nely <= 0 or nelz < 0:
            return "Nx, Ny, Nz must be positive."

        err = (
            self._check_forces(params)
            or self._check_regions(params)
            or self._check_supports(params)
            or self._check_force_duplicates(params)
            or self._check_materials(params)
        )
        return err

    def _check_duplicates(self, indices, keyfunc, msg):
        seen = {}
        for i in indices:
            k = keyfunc(i)
            if k in seen:
                return msg(seen[k], i)
            seen[k] = i

    def _check_forces(self, params):
        pf = params["Forces"]
        ps = params.get("Supports", {})

        if not any(d != "-" for d in pf["fidir"]):
            return "At least one input force must be active"

        if not any(d != "-" for d in pf["fodir"]) and not any(
            d != "-" for d in ps.get("sdim", [])
        ):
            return "At least one output force (for compliant mechanisms) or support (for rigid mechanisms) must be active"

    def _check_regions(self, params):
        pr = params.get("Region")
        if not pr:
            return

        idx = [i for i, s in enumerate(pr["rshape"]) if s != "-"]

        return self._check_duplicates(
            idx,
            lambda i: (
                pr["rshape"][i],
                pr["rradius"][i],
                pr["rx"][i],
                pr["ry"][i],
                pr["rz"][i],
            ),
            lambda a, b: f"Regions {a+1} and {b+1} are identical.",
        )

    def _check_supports(self, params):
        ps = params.get("Supports")
        if not ps:
            return

        idx = [i for i, s in enumerate(ps["sdim"]) if s != "-"]

        return self._check_duplicates(
            idx,
            lambda i: (ps["sx"][i], ps["sy"][i], ps["sz"][i], ps["sdim"][i]),
            lambda a, b: f"Supports {a+1} and {b+1} are identical.",
        )

    def _check_force_duplicates(self, params):
        pf = params["Forces"]

        err = self._check_duplicates(
            [i for i, d in enumerate(pf["fidir"]) if d != "-"],
            lambda i: (pf["fix"][i], pf["fiy"][i], pf["fiz"][i], pf["fidir"][i]),
            lambda a, b: f"Input forces {a+1} and {b+1} are identical.",
        )
        if err:
            return err

        return self._check_duplicates(
            [i for i, d in enumerate(pf["fodir"]) if d != "-"],
            lambda i: (pf["fox"][i], pf["foy"][i], pf["foz"][i], pf["fodir"][i]),
            lambda a, b: f"Output forces {a+1} and {b+1} are identical.",
        )

    def _check_materials(self, params):
        pm = params["Materials"]

        err = self._check_duplicates(
            range(len(pm["E"])),
            lambda i: (pm["E"][i], pm["nu"][i], pm.get("percent", [100])[i]),
            lambda a, b: f"Materials {a+1} and {b+1} are identical.",
        )
        if err:
            return err

        if len(pm["E"]) > 1 and sum(pm["percent"]) != 100:
            return "Material percentages don't sum up to 100%."

    def update_position_ranges(self):
        """Updates the maximum values for all position-related spin boxes."""
        nelx = self.dim_widget.nx.value()
        nely = self.dim_widget.ny.value()
        nelz = self.dim_widget.nz.value()

        # Update ranges for all regions
        for rw in self.regions_widget.inputs:
            rw["rx"].setMaximum(nelx)
            rw["ry"].setMaximum(nely)
            rw["rz"].setMaximum(nelz)
            rw["rradius"].setMaximum(
                min(nelx, nely, nelz) if nelz > 0 else min(nelx, nely)
            )

        # Update input forces
        for fw in self.forces_widget.inputs:
            if "fix" in fw:
                fw["fix"].setMaximum(nelx)
                fw["fiy"].setMaximum(nely)
                fw["fiz"].setMaximum(nelz)
            elif "fox" in fw:
                fw["fox"].setMaximum(nelx)
                fw["foy"].setMaximum(nely)
                fw["foz"].setMaximum(nelz)

        # Update supports
        for sw in self.supports_widget.inputs:
            sw["sx"].setMaximum(nelx)
            sw["sy"].setMaximum(nely)
            sw["sz"].setMaximum(nelz)

        # Update regions
        for rw in self.regions_widget.inputs:
            rw["rx"].setMaximum(nelx)
            rw["ry"].setMaximum(nely)
            rw["rz"].setMaximum(nelz)
            # Radius limit?
            max_dim = max(nelx, nely, nelz)
            rw["rradius"].setMaximum(max_dim)

    def scale_parameters(self):
        """Scales all dimensional and positional parameters by a given factor."""
        scale = self.dim_widget.scale.value()
        if scale == 1.0:
            self.status_bar.showMessage("Scale is 1.0, nothing to do.", 3000)
            return

        is_3d = self.dim_widget.nz.value() > 0
        axes = ["x", "y", "z"] if is_3d else ["x", "y"]

        # Track dimensions separately so they can be scaled BEFORE positions
        dims_to_scale = [self.dim_widget.nx, self.dim_widget.ny]
        if is_3d:
            dims_to_scale.append(self.dim_widget.nz)

        pos_to_validate = []
        pos_to_scale = []

        def register(widget, active, is_radius=False):
            if active:
                pos_to_validate.append(widget)
            pos_to_scale.append((widget, is_radius))

        # --- Gather parameters ---
        # 1Gather Regions
        for rw in self.regions_widget.inputs:
            active = rw["rshape"].currentText() != "-"
            for ax in axes:
                register(rw["r" + ax], active)
            # Original logic validated rradius regardless of active state
            register(rw["rradius"], True, True)

        # Gather Forces
        for fw in self.forces_widget.inputs:
            if "fidir" in fw:
                active = fw["fidir"].currentText() != "-"
                for ax in axes:
                    register(fw["fi" + ax], active)
            elif "fodir" in fw:
                active = fw["fodir"].currentText() != "-"
                for ax in axes:
                    register(fw["fo" + ax], active)

        # Gather Supports
        for sw in self.supports_widget.inputs:
            active = sw["sdim"].currentText() != "-"
            for ax in axes:
                register(sw["s" + ax], active)

        # --- Validation ---
        proceed_impossible, warn_needed = False, False

        for w in dims_to_scale + pos_to_validate:
            val = w.value() * scale
            if (val < 1 or val > 1000) and w.value() > 0:
                proceed_impossible = True
            elif abs(val - round(val)) > 1e-6:
                warn_needed = True

        if proceed_impossible:
            QMessageBox.critical(
                self, "Scaling Error", "Scaling would lead position(s) out of range."
            )
            return

        if warn_needed:
            reply = QMessageBox.question(
                self,
                "Scaling Warning",
                "Scaling would loss initial proportions due to rounding(s). Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # --- Perform Scaling ---
        # Temporarily block signals to prevent multiple replots
        self.block_all_parameter_signals(True)

        for w in dims_to_scale:
            w.setValue(round(w.value() * scale))

        if scale > 1.0:
            self.update_position_ranges()  # Update max ranges before scaling positions otherwise they might get clamped

        for w, is_radius in pos_to_scale:
            new_val = round(w.value() * scale)
            if is_radius:
                new_val = max(1, new_val)
            w.setValue(new_val)

        if scale < 1.0:
            self.update_position_ranges()  # Update max ranges after scaling positions otherwise values might be clamped before scaling

        self.block_all_parameter_signals(False)

        # Manually trigger a single, final update
        self.on_parameter_changed()
        self.status_bar.showMessage(
            f"All parameters scaled by a factor of {scale}.", 3000
        )

    def block_all_parameter_signals(self, block: bool):
        """Helper to block or unblock signals for all parameter widgets."""
        all_widgets = [
            self.dim_widget.nx,
            self.dim_widget.ny,
            self.dim_widget.nz,
            self.dim_widget.volfrac,
            self.materials_widget.mat_init_type,
            self.optimizer_widget.opt_ft,
            self.optimizer_widget.opt_fr,
            self.optimizer_widget.opt_p,
            self.optimizer_widget.opt_eta,
            self.optimizer_widget.opt_max_change,
            self.optimizer_widget.opt_n_it,
            self.optimizer_widget.opt_solver,
        ]
        for w in all_widgets:
            w.blockSignals(block)
        for group in (
            self.regions_widget.inputs
            + self.forces_widget.inputs
            + self.supports_widget.inputs
            + self.materials_widget.inputs
        ):
            for w in group.values():
                w.blockSignals(block)
