# app/ui/main_window.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Main window for the Topopt Comec application using PySide6.

import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core import initializers

# import mcubes
from app.ui import exporters

from .icons import icons
from .themes import DARK_THEME_STYLESHEET, LIGHT_THEME_STYLESHEET
from .widgets import (
    CollapsibleSection,
    DimensionsWidget,
    DisplacementWidget,
    FooterWidget,
    ForcesWidget,
    HeaderWidget,
    MaterialWidget,
    OptimizerWidget,
    PresetWidget,
    RegionsWidget,
    SupportWidget,
)
from .workers import DisplacementWorker, OptimizerWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Topopt Comec - Topology Optimization for Compliant Mechanisms"
        )
        self.setGeometry(100, 100, 1280, 720)

        # Consolidate duplicate variable declarations
        self.xPhys = None
        self.u = None
        self.last_params = {}
        self.current_theme = "dark"
        self.displacement_worker = None
        self.worker = None  # To hold the optimizer worker

        self.set_theme(self.current_theme)

        self.presets = {}
        self.presets_file = "presets.json"

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.plot_frame = QFrame()
        plot_layout = QVBoxLayout(self.plot_frame)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.splitter.addWidget(self.plot_frame)

        self.is_displaying_deformation = False
        self.last_displayed_frame_data = None

        self.create_control_panel()
        self.splitter.addWidget(self.control_panel_frame)
        self.splitter.setSizes([800, 480])

        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.load_presets()

        default_preset_name = "ForceInverter_2Sup_2D"
        if default_preset_name in self.presets:
            self.preset.presets_combo.setCurrentText(default_preset_name)
            self.on_preset_selected()  # This applies the preset and replots
        else:
            # Fallback in case the default preset isn't found
            print(f"Warning: Default preset '{default_preset_name}' not found.")
            self.last_params = self.gather_parameters()
            self.replot()

    #################
    # CONTROL PANEL #
    #################

    def create_control_panel(self):
        """Creates the right-hand side control panel with all settings."""
        self.control_panel_frame = QFrame()
        self.control_panel_frame.setFixedWidth(350)
        panel_layout = QVBoxLayout(self.control_panel_frame)

        # Header
        self.header = self.create_header()
        panel_layout.addWidget(self.header)

        # Preset
        self.preset = self.create_preset_section()
        panel_layout.addWidget(self.preset)

        # Parameters sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sections_layout = QVBoxLayout(scroll_widget)
        self.sections_layout.setAlignment(Qt.AlignTop)
        scroll_widget.setLayout(self.sections_layout)
        scroll_area.setWidget(scroll_widget)
        panel_layout.addWidget(scroll_area)

        self.sections = {}
        self.sections["dimensions"] = self.create_dimensions_section()
        self.sections["regions"] = self.create_regions_section()
        self.sections["forces"] = self.create_forces_section()
        self.sections["supports"] = self.create_supports_section()
        self.sections["material"] = self.create_material_section()
        self.sections["optimizer"] = self.create_optimizer_section()
        self.sections["displacement"] = self.create_displacement_section()

        for section in self.sections.values():
            self.sections_layout.addWidget(section)

        # Footer
        self.footer = self.create_footer()
        panel_layout.addWidget(self.footer)

    ############
    # SECTIONS #
    ############

    def create_header(self):
        """Creates the header widget and connects its signals."""
        header_widget = HeaderWidget()

        # Connect the signals from the widget's public buttons to MainWindow's handlers
        header_widget.info_button.clicked.connect(self.open_github_link)
        header_widget.theme_button.clicked.connect(self.toggle_theme)

        return header_widget

    def create_preset_section(self):
        """Creates the preset widget and connects its signals."""
        preset_widget = PresetWidget()
        preset_widget.presets_combo.activated.connect(self.on_preset_selected)
        preset_widget.save_preset_button.clicked.connect(self.save_new_preset)
        preset_widget.delete_preset_button.clicked.connect(self.delete_selected_preset)
        return preset_widget

    def create_dimensions_section(self):
        """Creates the first section for dimensions and volume fraction."""
        self.dim_widget = DimensionsWidget()
        section = CollapsibleSection("🔲 Dimensions", self.dim_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        self.dim_widget.nx.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.ny.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.nz.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.volfrac.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.scale_button.clicked.connect(self.scale_parameters)
        self.dim_widget.nx.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.ny.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self.update_position_ranges)
        return section

    def create_regions_section(self):
        """Creates the second section for regions parameters."""
        self.regions_widget = RegionsWidget()
        section = CollapsibleSection("⚫ Regions", self.regions_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        for region_group in self.regions_widget.inputs:
            region_group["rshape"].currentIndexChanged.connect(
                self.on_parameter_changed
            )
            region_group["rstate"].currentIndexChanged.connect(
                self.on_parameter_changed
            )
            region_group["rradius"].valueChanged.connect(self.on_parameter_changed)
            region_group["rx"].valueChanged.connect(self.on_parameter_changed)
            region_group["ry"].valueChanged.connect(self.on_parameter_changed)
            region_group["rz"].valueChanged.connect(self.on_parameter_changed)
        return section

    def create_forces_section(self):
        """Creates the third section for forces parameters."""
        self.forces_widget = ForcesWidget()
        section = CollapsibleSection("💪 Forces", self.forces_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)

        # 3. Connect the signals from the widgets INSIDE the ForcesWidget
        for force_group in self.forces_widget.inputs:
            if "fix" in force_group:
                force_group["fix"].valueChanged.connect(self.on_parameter_changed)
                force_group["fiy"].valueChanged.connect(self.on_parameter_changed)
                force_group["fiz"].valueChanged.connect(self.on_parameter_changed)
                force_group["fidir"].currentIndexChanged.connect(
                    self.on_parameter_changed
                )
                force_group["finorm"].valueChanged.connect(self.on_parameter_changed)
            elif "fox" in force_group:
                force_group["fox"].valueChanged.connect(self.on_parameter_changed)
                force_group["foy"].valueChanged.connect(self.on_parameter_changed)
                force_group["foz"].valueChanged.connect(self.on_parameter_changed)
                force_group["fodir"].currentIndexChanged.connect(
                    self.on_parameter_changed
                )
                force_group["fonorm"].valueChanged.connect(self.on_parameter_changed)

        return section

    def create_supports_section(self):
        """Creates the fourth section for supports parameters."""
        self.supports_widget = SupportWidget()
        section = CollapsibleSection("🔺 Supports", self.supports_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)

        for support_input_group in self.supports_widget.inputs:
            support_input_group["sx"].valueChanged.connect(self.on_parameter_changed)
            support_input_group["sy"].valueChanged.connect(self.on_parameter_changed)
            support_input_group["sz"].valueChanged.connect(self.on_parameter_changed)
            support_input_group["sdim"].currentIndexChanged.connect(
                self.on_parameter_changed
            )

        return section

    def create_material_section(self):
        """Creates the fifth section for material properties."""
        self.material_widget = MaterialWidget()
        section = CollapsibleSection("🧱 Material", self.material_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        self.material_widget.mat_E.valueChanged.connect(self.on_parameter_changed)
        self.material_widget.mat_nu.valueChanged.connect(self.on_parameter_changed)
        self.material_widget.mat_color.clicked.connect(self.replot)
        self.material_widget.mat_init_type.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        return section

    def create_optimizer_section(self):
        """Creates the sixth section for optimization parameters."""
        self.optimizer_widget = OptimizerWidget()
        section = CollapsibleSection("💻 Optimizer", self.optimizer_widget)
        self.optimizer_widget.opt_ft.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_fr.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_p.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_max_change.valueChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_n_it.valueChanged.connect(self.on_parameter_changed)
        return section

    def create_displacement_section(self):
        """Creates the seventh section for displacement animation parameters."""
        self.displacement_widget = DisplacementWidget()
        section = CollapsibleSection("↔️ Displacement", self.displacement_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        section.visibility_button.setEnabled(False)  # Disabled until a result is ready
        self.displacement_widget.run_disp_button.setEnabled(
            False
        )  # Disabled until a result is ready
        section.visibility_button.setToolTip(
            "Preview displacement vectors on the main plot"
        )
        section.visibility_button.setChecked(False)

        section.visibility_button.toggled.connect(self.replot)
        self.displacement_widget.run_disp_button.clicked.connect(self.run_displacement)
        self.displacement_widget.stop_disp_button.clicked.connect(
            self.stop_displacement
        )
        self.displacement_widget.reset_disp_button.clicked.connect(
            self.reset_displacement_view
        )
        self.displacement_widget.mov_disp.valueChanged.connect(
            self.on_displacement_preview_changed
        )

        return section

    def on_visibility_toggled(self, checked):
        """Handles the toggling of any visibility button."""
        button = self.sender()  # method gives the specific button that was clicked.
        if not button:
            return

        if checked:
            button.setIcon(icons.get("eye_open"))
            button.setToolTip("Element is visible. Click to hide.")
        else:
            button.setIcon(icons.get("eye_closed"))
            button.setToolTip("Element is hidden. Click to show.")

        self.replot()

    def create_footer(self):
        """Creates the footer widget and connects its signals."""
        footer_widget = FooterWidget()
        footer_widget.create_button.clicked.connect(self.run_optimization)
        footer_widget.stop_button.clicked.connect(self.stop_optimization)
        footer_widget.binarize_button.clicked.connect(self.on_binarize_clicked)
        footer_widget.save_png_action.triggered.connect(self.save_as_png)
        footer_widget.save_vti_action.triggered.connect(self.save_as_vti)
        footer_widget.save_stl_action.triggered.connect(self.save_as_stl)
        return footer_widget

    ##############
    # PARAMETERS #
    ##############

    def gather_parameters(self):
        """Collects all parameters from the UI controls into a dictionary."""
        params = {}
        # Dimensions
        nelx, nely, nelz = (
            self.dim_widget.nx.value(),
            self.dim_widget.ny.value(),
            self.dim_widget.nz.value(),
        )
        params["nelxyz"] = [nelx, nely, nelz]
        params["volfrac"] = self.dim_widget.volfrac.value()

        # Regions
        params["rshape"], params["rstate"], params["rradius"] = [], [], []
        params["rx"], params["ry"], params["rz"] = [], [], []
        for rw in self.regions_widget.inputs:
            params["rshape"].append(rw["rshape"].currentText())
            params["rstate"].append(rw["rstate"].currentText())
            params["rradius"].append(rw["rradius"].value())
            params["rx"].append(rw["rx"].value())
            params["ry"].append(rw["ry"].value())
            params["rz"].append(rw["rz"].value())

        # Forces
        params["fix"], params["fiy"], params["fiz"] = [], [], []
        params["fidir"], params["finorm"] = [], []
        params["fox"], params["foy"], params["foz"] = [], [], []
        params["fodir"], params["fonorm"] = [], []
        for fw in self.forces_widget.inputs:
            if "fix" in fw:  # Input force
                params["fix"].append(fw["fix"].value())
                params["fiy"].append(fw["fiy"].value())
                params["fiz"].append(fw["fiz"].value())
                params["fidir"].append(fw["fidir"].currentText())
                params["finorm"].append(fw["finorm"].value())
            elif "fox" in fw:  # Output force
                params["fox"].append(fw["fox"].value())
                params["foy"].append(fw["foy"].value())
                params["foz"].append(fw["foz"].value())
                params["fodir"].append(fw["fodir"].currentText())
                params["fonorm"].append(fw["fonorm"].value())

        # Supports
        params["sx"], params["sy"], params["sz"] = [], [], []
        params["sdim"] = []
        for sw in self.supports_widget.inputs:
            params["sx"].append(sw["sx"].value())
            params["sy"].append(sw["sy"].value())
            params["sz"].append(sw["sz"].value())
            params["sdim"].append(sw["sdim"].currentText())

        # Material
        params["E"] = self.material_widget.mat_E.value()
        params["nu"] = self.material_widget.mat_nu.value()
        params["init_type"] = self.material_widget.mat_init_type.currentIndex()

        # Optimizer
        params["filter_type"] = (
            "Sensitivity"
            if self.optimizer_widget.opt_ft.currentIndex() == 0
            else "Density"
        )
        params["filter_radius_min"] = self.optimizer_widget.opt_fr.value()
        params["penal"] = self.optimizer_widget.opt_p.value()
        params["max_change"] = self.optimizer_widget.opt_max_change.value()
        params["n_it"] = self.optimizer_widget.opt_n_it.value()

        # Movement
        params["disp_factor"] = self.displacement_widget.mov_disp.value()
        params["disp_iterations"] = self.displacement_widget.mov_iter.value()

        return params

    def on_parameter_changed(self):
        """React when a parameter is changed."""  #
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
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.sections["displacement"].visibility_button.setEnabled(False)

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
        p1_norm = copy.deepcopy(params1)
        p2_norm = copy.deepcopy(params2)

        def normalize_params(p):
            if "nelxyz" in p:
                is_2d = len(p["nelxyz"]) < 3 or p["nelxyz"][2] == 0.0
                if is_2d:
                    p["nelxyz"] = p["nelxyz"][:2]
            # --- Normalize Regions ---
            if "rshape" in p:
                zipped_regions = zip(
                    p.get("rshape", []),
                    p.get("rstate", []),
                    p.get("rradius", []),
                    p.get("rx", []),
                    p.get("ry", []),
                    p.get("rz", []) if not is_2d else [0] * len(p.get("rx", [])),
                )
                region_list = list(zipped_regions)
                active_regions = [r for r in region_list if r[0] != "-"]
                if active_regions:
                    rshape, rstate, rradius, rx, ry, rz = list(zip(*active_regions))
                    (
                        p["rshape"],
                        p["rstate"],
                        p["rradius"],
                        p["rx"],
                        p["ry"],
                        p["rz"],
                    ) = (
                        list(rshape),
                        list(rstate),
                        list(rradius),
                        list(rx),
                        list(ry),
                        list(rz),
                    )
                else:
                    for key in ["rshape", "rstate", "rradius", "rx", "ry", "rz"]:
                        p.pop(key, None)  # pop them, not just empty them
                if is_2d and "rz" in p:
                    p.pop("rz")
            # --- Normalize Supports ---
            if "sdim" in p:
                zipped_supports = zip(
                    p.get("sx", []),
                    p.get("sy", []),
                    p.get("sz", []) if not is_2d else [0] * len(p.get("sx", [])),
                    p.get("sdim", []),
                )
                active_supports = [s for s in zipped_supports if s[3] != "-"]
                if active_supports:
                    sx, sy, sz, sdim = list(zip(*active_supports))
                    p["sx"], p["sy"], p["sz"], p["sdim"] = (
                        list(sx),
                        list(sy),
                        list(sz),
                        list(sdim),
                    )
                else:  # Should not happen as at least one support is required
                    p["sx"], p["sy"], p["sz"], p["sdim"] = [], [], [], []
                if is_2d and "sz" in p:
                    p.pop("sz")
            # --- Normalize Forces ---
            if "fidir" in p:
                zipped_forces = zip(
                    p.get("fix", []),
                    p.get("fiy", []),
                    p.get("fiz", []) if not is_2d else [0] * len(p.get("fix", [])),
                    p.get("fidir", []),
                    p.get("finorm", []),
                )
                force_list = list(zipped_forces)
                # Keep the input force, and any output forces that are active.
                active_forces = [force_list[0]] + [
                    f for f in force_list[1:] if f[3] != "-"
                ]
                if active_forces:
                    fx, fy, fz, fdir, fnorm = list(zip(*active_forces))
                    p["fix"], p["fiy"], p["fiz"], p["fidir"], p["finorm"] = (
                        list(fx),
                        list(fy),
                        list(fz),
                        list(fdir),
                        list(fnorm),
                    )
                else:  # Should not happen as one input force is required
                    p["fix"], p["fiy"], p["fiz"], p["fidir"], p["finorm"] = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                if is_2d and "fiz" in p:
                    p.pop("fiz")
            if "fodir" in p:
                zipped_forces = zip(
                    p.get("fox", []),
                    p.get("foy", []),
                    p.get("foz", []) if not is_2d else [0] * len(p.get("fox", [])),
                    p.get("fodir", []),
                    p.get("fonorm", []),
                )
                force_list = list(zipped_forces)
                # Keep the input force, and any output forces that are active.
                active_forces = [force_list[0]] + [
                    f for f in force_list[1:] if f[3] != "-"
                ]
                if active_forces:
                    fx, fy, fz, fdir, fnorm = list(zip(*active_forces))
                    p["fox"], p["foy"], p["foz"], p["fodir"], p["fonorm"] = (
                        list(fx),
                        list(fy),
                        list(fz),
                        list(fdir),
                        list(fnorm),
                    )
                else:  # Should not happen as one input force is required
                    p["fox"], p["foy"], p["foz"], p["fodir"], p["fonorm"] = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                if is_2d and "foz" in p:
                    p.pop("foz")

            return p

        # Normalize both dictionaries
        p1_norm = normalize_params(p1_norm)
        p2_norm = normalize_params(p2_norm)

        # Now, compare the normalized, canonical versions
        return json.dumps(p1_norm, sort_keys=True) == json.dumps(
            p2_norm, sort_keys=True
        )

    def validate_parameters(self, p):
        """Checks for common input errors."""
        nelx, nely, nelz = p["nelxyz"]
        if nelx <= 0 or nely <= 0 or nelz < 0:
            return "Nx, Ny, Nz must be positive."
        if not any(d != "-" for d in p["fidir"]):
            print("At least one input force must be active")
        if not any(d != "-" for d in p["fodir"]):
            print("At least one output force must be active")
        has_support = any(d != "-" for d in p["sdim"])
        if not has_support:
            return "At least one support must be defined."
        return None

    def update_position_ranges(self):
        """
        Updates the maximum values for all position-related spin boxes
        based on the current Nx, Ny, and Nz values.
        """
        # Get the current maximums from the dimensions widget
        # The check for self.dim_widget handles the initial app startup
        if not hasattr(self, "dim_widget"):
            return

        # Get the current maximums from the dimensions widget
        nelx = self.dim_widget.nx.value()
        nely = self.dim_widget.ny.value()
        nelz = self.dim_widget.nz.value()

        # Update ranges for all regions
        for region_group in self.regions_widget.inputs:
            region_group["rx"].setMaximum(nelx)
            region_group["ry"].setMaximum(nely)
            region_group["rz"].setMaximum(nelz)
            region_group["rradius"].setMaximum(
                min(nelx, nely, nelz) if nelz > 0 else min(nelx, nely)
            )

        # Update ranges for all forces
        for force_group in self.forces_widget.inputs:
            if "fix" in force_group:  # Input force
                force_group["fix"].setMaximum(nelx)
                force_group["fiy"].setMaximum(nely)
                force_group["fiz"].setMaximum(nelz)
            elif "fox" in force_group:  # Output force
                force_group["fox"].setMaximum(nelx)
                force_group["foy"].setMaximum(nely)
                force_group["foz"].setMaximum(nelz)

        # Update ranges for all supports
        for support_group in self.supports_widget.inputs:
            support_group["sx"].setMaximum(nelx)
            support_group["sy"].setMaximum(nely)
            support_group["sz"].setMaximum(nelz)

    def scale_parameters(self):
        """Scales all dimensional and positional parameters by a given factor."""
        scale = self.dim_widget.scale.value()
        is_3d = self.dim_widget.nz.value() > 0

        if scale == 1.0:
            self.status_bar.showMessage("Scale is 1.0, nothing to do.", 3000)
            return

        def check(value, scale):
            scaled_val = value * scale
            if (scaled_val < 1 or scaled_val > 1000) and value > 0:
                return True, False
            if (
                abs(scaled_val - round(scaled_val)) > 1e-6
            ):  # Check if it's not an integer
                return False, True
            return False, False

        proceed_impossible, warn_needed = False, False

        # Check dimensions
        for dim_widget in [self.dim_widget.nx, self.dim_widget.ny] + (
            [self.dim_widget.nz] if is_3d else []
        ):
            pi, wn = check(dim_widget.value(), scale)
            proceed_impossible |= pi
            warn_needed |= wn

        # Check regions
        for region_group in self.regions_widget.inputs:
            if region_group["rshape"].currentText() != "-":
                for key in ["rx", "ry"] + (["rz"] if is_3d else []):
                    pi, wn = check(region_group[key].value(), scale)
                    proceed_impossible |= pi
                    warn_needed |= wn
            pi, wn = check(region_group["rradius"].value(), scale)
            proceed_impossible |= pi
            warn_needed |= wn

        # Check forces
        for force_group in self.forces_widget.inputs:
            if "fidir" in force_group:
                if force_group["fidir"].currentText() != "-":
                    for key in ["fix", "fiy"] + (["fiz"] if is_3d else []):
                        pi, wn = check(force_group[key].value(), scale)
                        proceed_impossible |= pi
                        warn_needed |= wn
            elif "fodir" in force_group:
                if force_group["fodir"].currentText() != "-":
                    for key in ["fox", "foy"] + (["foz"] if is_3d else []):
                        pi, wn = check(force_group[key].value(), scale)
                        proceed_impossible |= pi
                        warn_needed |= wn

        # Check supports
        for support_group in self.supports_widget.inputs:
            if support_group["sdim"].currentText() != "-":
                for key in ["sx", "sy"] + (["sz"] if is_3d else []):
                    pi, wn = check(support_group[key].value(), scale)
                    proceed_impossible |= pi
                    warn_needed |= wn

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

        self.dim_widget.nx.setValue(round(self.dim_widget.nx.value() * scale))
        self.dim_widget.ny.setValue(round(self.dim_widget.ny.value() * scale))
        if is_3d:
            self.dim_widget.nz.setValue(round(self.dim_widget.nz.value() * scale))

        if scale > 1.0:
            self.update_position_ranges()  # Update max ranges before scaling positions otherwise they might get clamped

        for region_group in self.regions_widget.inputs:
            region_group["rx"].setValue(round(region_group["rx"].value() * scale))
            region_group["ry"].setValue(round(region_group["ry"].value() * scale))
            if is_3d:
                region_group["rz"].setValue(round(region_group["rz"].value() * scale))
            region_group["rradius"].setValue(
                max(1, round(region_group["rradius"].value() * scale))
            )

        for force_group in self.forces_widget.inputs:
            if "fidir" in force_group:
                force_group["fix"].setValue(round(force_group["fix"].value() * scale))
                force_group["fiy"].setValue(round(force_group["fiy"].value() * scale))
                if is_3d:
                    force_group["fiz"].setValue(
                        round(force_group["fiz"].value() * scale)
                    )
            elif "fodir" in force_group:
                force_group["fox"].setValue(round(force_group["fox"].value() * scale))
                force_group["foy"].setValue(round(force_group["foy"].value() * scale))
                if is_3d:
                    force_group["foz"].setValue(
                        round(force_group["foz"].value() * scale)
                    )

        for support_group in self.supports_widget.inputs:
            support_group["sx"].setValue(round(support_group["sx"].value() * scale))
            support_group["sy"].setValue(round(support_group["sy"].value() * scale))
            if is_3d:
                support_group["sz"].setValue(round(support_group["sz"].value() * scale))

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
        # A helper to make the code cleaner. Add all your widgets to this list.
        all_widgets = [
            self.dim_widget.nx,
            self.dim_widget.ny,
            self.dim_widget.nz,
            self.dim_widget.volfrac,
            self.material_widget.mat_E,
            self.material_widget.mat_nu,
            self.material_widget.mat_init_type,
            self.optimizer_widget.opt_ft,
            self.optimizer_widget.opt_fr,
            self.optimizer_widget.opt_p,
            self.optimizer_widget.opt_max_change,
            self.optimizer_widget.opt_n_it,
        ]
        for w in all_widgets:
            w.blockSignals(block)
        for group in (
            self.regions_widget.inputs
            + self.forces_widget.inputs
            + self.supports_widget.inputs
        ):
            for w in group.values():
                w.blockSignals(block)

    ################
    # OPTIMIZATION #
    ################

    def run_optimization(self):
        """Starts the optimization process based on current parameters, and gives live updates."""
        error = self.validate_parameters(self.last_params)
        if error:
            QMessageBox.critical(self, "Input Error", error)
            return

        if self.is_displaying_deformation:
            self.reset_displacement_view()

        # Stop animation
        self.footer.stop_create_button_effect()

        # This creates the gray box that will be updated live
        self.replot()
        QApplication.processEvents()  # Force the UI to draw the initial state

        self.preset.setEnabled(False)
        self.footer.create_button.hide()
        self.footer.stop_button.setText(" Stop")
        self.footer.stop_button.setEnabled(True)
        self.footer.stop_button.show()
        self.footer.create_button.setEnabled(False)
        self.footer.binarize_button.setEnabled(False)
        self.footer.save_button.setEnabled(False)
        self.status_bar.showMessage("Starting optimization...")
        self.progress_bar.setRange(0, self.last_params["n_it"])
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = OptimizerWorker(self.last_params)
        self.worker.progress.connect(self.update_optimization_progress)
        self.worker.frameReady.connect(self.update_optimization_plot)
        self.worker.finished.connect(self.handle_optimization_results)
        self.worker.error.connect(self.handle_optimization_error)
        self.worker.start()

    def stop_optimization(self):
        """Requests the running optimizer worker to stop."""
        if self.worker:
            self.status_bar.showMessage("Stopping optimization...")
            self.footer.stop_button.setText("Stopping...")
            self.footer.stop_button.setEnabled(False)  # Prevent multiple clicks
            self.worker.request_stop()

    def update_optimization_progress(self, iteration, objective, change):
        """Updates the progress bar and status message during optimization."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(
            f"It: {iteration}, Obj: {objective:.4f}, Change: {change:.4f}"
        )

    def update_optimization_plot(self, xPhys_frame):
        """Updates the plot with an intermediate frame from the optimizer."""
        # Ensure a plot exist to update
        if not self.figure.get_axes():
            return

        ax = self.figure.get_axes()[0]

        # Get dimensions and update the image data
        is_3d = self.last_params["nelxyz"][2] > 0 if self.last_params else False
        if is_3d:
            self.plot_material(ax, is_3d=is_3d, xPhys_data=xPhys_frame)
            self.redraw_non_material_layers(ax, is_3d=True)
        else:
            if not ax.images:
                return
            im = ax.images[0]  # The imshow object is the first image on the axes
            nelx, nely = self.last_params["nelxyz"][:2]
            im.set_array(xPhys_frame.reshape((nelx, nely)).T)

        self.canvas.draw()

    def handle_optimization_results(self, result):
        """Handles the results after optimization finishes successfully."""
        self.xPhys, self.u = result
        self.status_bar.showMessage("Optimization finished successfully.", 5000)
        self.preset.setEnabled(True)
        self.footer.stop_button.hide()
        self.footer.create_button.show()
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.footer.binarize_button.setEnabled(True)
        self.footer.save_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.sections["displacement"].visibility_button.setEnabled(True)
        self.replot()

    def handle_optimization_error(self, error_msg):
        """Handles any errors that occur during optimization."""
        self.status_bar.showMessage("Optimization failed.", 5000)
        self.preset.setEnabled(True)
        self.footer.stop_button.hide()
        self.footer.create_button.show()
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.footer.binarize_button.setEnabled(True)
        self.footer.save_button.setEnabled(True)
        QMessageBox.critical(self, "Runtime Error", error_msg)

    ################
    # DISPLACEMENT #
    ################

    def run_displacement(self):
        """Starts the displacement animation based on the last optimization result, and gives live updates."""
        if self.xPhys is None or self.u is None:
            QMessageBox.warning(
                self,
                "Displacement Error",
                "You must run a successful optimization before analyzing movement.",
            )
            return

        self.is_displaying_deformation = True
        self.last_displayed_frame_data = None
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.stop_disp_button
        )
        self.footer.create_button.setEnabled(False)

        QApplication.processEvents()

        self.last_params = self.gather_parameters()
        if self.last_params["disp_iterations"] == 1:
            # Run single-frame logic directly
            self.status_bar.showMessage("Calculating single displacement frame...")
            QApplication.processEvents()  # Update UI
            from app.core.displacements import single_linear_displacement

            nelx, nely, nelz = self.last_params["nelxyz"]
            self.last_displayed_frame_data = single_linear_displacement(
                self.u, nelx, nely, nelz, self.last_params["disp_factor"]
            )
            self.replot()
            self.handle_displacement_finished("Single frame shown.")
            self.status_bar.showMessage("Single displacement plot shown.", 3000)
        else:
            self.footer.create_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.status_bar.showMessage("Starting displacement computation...")

            self.progress_bar.setRange(0, self.last_params["disp_iterations"] + 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            self.displacement_worker = DisplacementWorker(
                self.last_params, self.xPhys, self.u
            )
            self.displacement_worker.progress.connect(self.update_displacement_progress)
            self.displacement_worker.frameReady.connect(self.update_animation_frame)
            self.displacement_worker.finished.connect(self.handle_displacement_finished)
            self.displacement_worker.error.connect(self.handle_displacement_error)
            self.displacement_worker.start()

    def stop_displacement(self):
        """Requests the running displacement worker to stop."""
        if self.displacement_worker:
            self.displacement_widget.stop_disp_button.setText("Stopping...")
            self.displacement_widget.stop_disp_button.setEnabled(False)
            self.displacement_worker.request_stop()

    def reset_displacement_view(self):
        """Resets the plot to the original, undeformed optimizer result."""
        self.is_displaying_deformation = False
        self.last_displayed_frame_data = None
        self.xPhys_display = self.xPhys.copy()
        self.replot()  # Redraw the original view
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.run_disp_button
        )

    def update_displacement_progress(self, iteration):
        """Updates the progress bar and status message during displacement computation."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(
            f"Running non-linear displacement: step {iteration}..."
        )

    def update_animation_frame(self, frame_data):
        """Updates the plot with a new frame from the displacement animation."""
        # Safety checks to ensure a plot exists and parameters are available
        if not self.figure.get_axes() or not self.last_params:
            return
        ax = self.figure.get_axes()[0]
        nelx, nely, nelz = self.last_params["nelxyz"]
        is_3d = nelz > 0

        if is_3d:
            ax.clear()

            # Use the fast scatter plot with variable alpha for the "cloud" effect
            visible_elements_mask = frame_data > 0.01
            visible_indices = np.where(visible_elements_mask)[0]
            densities = frame_data[visible_indices]

            # Calculate coordinates for only the visible elements
            z = visible_indices // (nelx * nely)
            x = (visible_indices % (nelx * nely)) // nely
            y = visible_indices % nely

            # Create the RGBA color array where alpha = density
            colors = np.zeros((len(densities), 4))
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            colors[:, :3] = base_color_rgb
            colors[:, 3] = densities

            ax.scatter(
                x + 0.5,
                y + 0.5,
                z + 0.5,
                s=6000 / max(nelx, nely, nelz),
                marker="s",  # Square markers to mimic voxels
                c=colors,
                alpha=None,
            )  # Alpha is now controlled by the 'c' array

            self.redraw_non_material_layers(ax, is_3d=True)

        else:
            if not ax.images:
                return
            im = ax.images[0]
            im.set_array(frame_data.reshape((nelx, nely)).T)

        # Redraw the canvas to show the changes
        self.canvas.draw()

    def on_displacement_preview_changed(self):
        """Triggers a replot if the preview is active when displacement factor changes."""
        if self.sections["displacement"].visibility_button.isChecked():
            self.replot()

    def handle_displacement_finished(self, message):
        """Handles the results after displacement computation finishes successfully."""
        self.status_bar.showMessage(message, 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(False)
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.reset_disp_button
        )
        self.displacement_widget.stop_disp_button.setText(
            " Stop"
        )  # Reset text for next run
        self.displacement_widget.stop_disp_button.setEnabled(True)
        self.is_displaying_deformation = True

    def handle_displacement_error(self, error_msg):
        """Handles any errors that occur during displacement computation."""
        self.status_bar.showMessage("Displacements failed.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        QMessageBox.critical(self, "Displacement Runtime Error", error_msg)

    ########
    # Save #
    ########

    def save_result_as(self, file_type):
        """
        Save the current result as file_type in a result folder.
        The result folder is created if it does not exist.
        """
        if self.xPhys is None:
            return

        os.makedirs("results", exist_ok=True)

        # Base filename depending on preset
        if self.preset.presets_combo.currentText() != "Select a preset...":
            base_name = self.preset.presets_combo.currentText()
        else:
            base_name = (
                "result_3d" if self.last_params["nelxyz"][2] > 0 else "result_2d"
            )

        # File dialog config
        filters = {
            "png": ("Save as PNG", "Portable Network Graphics (*.png)"),
            "vti": ("Save as VTI", "VTK Image Data (*.vti)"),
            "stl": ("Save as STL", "STL File (*.stl)"),
        }

        window_title, extension_filter = filters[file_type]
        default_path = f"results/{base_name}.{file_type}"

        filepath, _ = QFileDialog.getSaveFileName(
            self, window_title, default_path, extension_filter
        )
        if not filepath:  # user canceled
            return

        try:
            if file_type == "png":
                self.figure.savefig(filepath, dpi=300, bbox_inches="tight")

            elif file_type == "vti":
                success, error_msg = exporters.save_as_vti(
                    self.xPhys, self.last_params["nelxyz"], filepath
                )
                if not success:
                    raise Exception(error_msg)

            elif file_type == "stl":
                success, error_msg = exporters.save_as_stl(
                    self.xPhys, self.last_params["nelxyz"], filepath
                )
                if not success:
                    raise Exception(error_msg)

            self.status_bar.showMessage(f"Result saved to {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the file:\n{e}")

    def save_as_png(self):
        """Function connected to the save as PNG button."""
        self.save_result_as("png")

    def save_as_vti(self):
        """Function connected to the save as VTI button."""
        self.save_result_as("vti")

    def save_as_stl(self):
        """Function connected to the save as STL button."""
        self.save_result_as("stl")

    ########
    # PLOT #
    ########

    def style_plot_default(self):
        """Sets the plot to a fixed white theme. Called only once."""
        self.figure.patch.set_facecolor("white")
        if self.figure.get_axes():
            ax = self.figure.get_axes()[0]
            ax.set_facecolor("white")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.tick_params(axis="x", colors="black")
            ax.tick_params(axis="y", colors="black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
        self.canvas.draw()

    def replot(self):
        """Redraws the plot canvas, intelligently showing or hiding each layer based on the state of the visibility buttons."""
        if not self.last_params:
            return  # Do nothing if triggerd in sections initialization
        self.figure.clear()
        self.figure.patch.set_facecolor("white")
        nelx, nely, nelz = self.last_params["nelxyz"]
        is_3d = nelz > 0
        if is_3d:
            ax = self.figure.add_subplot(111, projection="3d", facecolor="white")
        else:
            ax = self.figure.add_subplot(111, facecolor="white")

        # Layer 1: The Main Result (Material)
        if (
            self.is_displaying_deformation
            and self.last_displayed_frame_data is not None
        ):
            if self.last_params["disp_iterations"] == 1:  # Single-frame grid plot
                if is_3d:
                    # Compute original element centers
                    nel = nelx * nely * nelz
                    visible_indices = np.arange(nel)  # all of them
                    z_idx = visible_indices // (nelx * nely)
                    x_idx = (visible_indices % (nelx * nely)) // nely
                    y_idx = visible_indices % nely

                    # Compute displaced centers from node positions
                    X, Y, Z = self.last_displayed_frame_data  # displaced node coords

                    # Take mean of the 8 node positions for each voxel center
                    cx = (
                        X[x_idx, y_idx, z_idx]
                        + X[x_idx + 1, y_idx, z_idx]
                        + X[x_idx, y_idx + 1, z_idx]
                        + X[x_idx + 1, y_idx + 1, z_idx]
                        + X[x_idx, y_idx, z_idx + 1]
                        + X[x_idx + 1, y_idx, z_idx + 1]
                        + X[x_idx, y_idx + 1, z_idx + 1]
                        + X[x_idx + 1, y_idx + 1, z_idx + 1]
                    ) / 8.0

                    cy = (
                        Y[x_idx, y_idx, z_idx]
                        + Y[x_idx + 1, y_idx, z_idx]
                        + Y[x_idx, y_idx + 1, z_idx]
                        + Y[x_idx + 1, y_idx + 1, z_idx]
                        + Y[x_idx, y_idx, z_idx + 1]
                        + Y[x_idx + 1, y_idx, z_idx + 1]
                        + Y[x_idx, y_idx + 1, z_idx + 1]
                        + Y[x_idx + 1, y_idx + 1, z_idx + 1]
                    ) / 8.0

                    cz = (
                        Z[x_idx, y_idx, z_idx]
                        + Z[x_idx + 1, y_idx, z_idx]
                        + Z[x_idx, y_idx + 1, z_idx]
                        + Z[x_idx + 1, y_idx + 1, z_idx]
                        + Z[x_idx, y_idx, z_idx + 1]
                        + Z[x_idx + 1, y_idx, z_idx + 1]
                        + Z[x_idx, y_idx + 1, z_idx + 1]
                        + Z[x_idx + 1, y_idx + 1, z_idx + 1]
                    ) / 8.0

                    # Colors with alpha = density
                    colors = np.zeros((nel, 4))
                    colors[:, :3] = to_rgb(self.material_widget.mat_color.get_color())
                    colors[:, 3] = self.xPhys

                    # Scatter plot of displaced centers
                    ax.scatter(
                        cx,
                        cy,
                        cz,
                        s=6000 / max(nelx, nely, nelz),
                        marker="s",
                        c=colors,
                        alpha=None,
                    )

                    ax.set_box_aspect([nelx, nely, nelz])
                else:
                    X, Y = self.last_displayed_frame_data
                    ax.pcolormesh(
                        X,
                        Y,
                        -self.xPhys.reshape((nelx, nely)),
                        cmap="gray",
                        shading="auto",
                    )
            # Multi-iteration displacement handled in update_animation_frame
        else:
            if self.sections["material"].visibility_button.isChecked():
                if self.xPhys is None:
                    p = self.last_params
                    # Initialize xPhys
                    active_iforces_indices = [
                        i
                        for i in range(len(p["fidir"]))
                        if np.array(p["fidir"])[i] != "-"
                    ]
                    active_oforces_indices = [
                        i
                        for i in range(len(p["fodir"]))
                        if np.array(p["fodir"])[i] != "-"
                    ]
                    active_supports_indices = [
                        i
                        for i in range(len(p["sdim"]))
                        if np.array(p["sdim"])[i] != "-"
                    ]
                    fix_active = np.array(p["fix"])[active_iforces_indices]
                    fiy_active = np.array(p["fiy"])[active_iforces_indices]
                    fox_active = np.array(p["fox"])[active_oforces_indices]
                    foy_active = np.array(p["foy"])[active_oforces_indices]
                    sx_active = np.array(p["sx"])[active_supports_indices]
                    sy_active = np.array(p["sy"])[active_supports_indices]
                    all_x = np.concatenate([fix_active, fox_active, sx_active])
                    all_y = np.concatenate([fiy_active, foy_active, sy_active])
                    if is_3d:
                        fiz_active = np.array(p["fiz"])[active_iforces_indices]
                        foz_active = np.array(p["foz"])[active_oforces_indices]
                        sz_active = np.array(p["sz"])[active_supports_indices]
                    all_z = (
                        np.concatenate([fiz_active, foz_active, sz_active])
                        if is_3d
                        else np.array([0] * len(all_x))
                    )
                    self.xPhys = initializers.initialize_material(
                        p["init_type"],
                        p["volfrac"],
                        nelx,
                        nely,
                        nelz,
                        all_x,
                        all_y,
                        all_z,
                    )
                    # Add regions if specified
                    for i, shape in enumerate(p["rshape"]):
                        if shape == "-":
                            continue
                        x_min, x_max = max(0, int(p["rx"][i] - p["rradius"][i])), min(
                            nelx, int(p["rx"][i] + p["rradius"][i]) + 1
                        )
                        y_min, y_max = max(0, int(p["ry"][i] - p["rradius"][i])), min(
                            nely, int(p["ry"][i] + p["rradius"][i]) + 1
                        )
                        if is_3d:
                            z_min, z_max = max(
                                0, int(p["rz"][i] - p["rradius"][i])
                            ), min(nelz, int(p["rz"][i] + p["rradius"][i]) + 1)

                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        if is_3d:
                            idx_z = np.arange(z_min, z_max)

                        if p["rshape"][i] == "□":  # Square/Cube
                            if len(idx_x) > 0 and len(idx_y) > 0:
                                if is_3d and len(idx_z) > 0:
                                    xx, yy, zz = np.meshgrid(
                                        idx_x, idx_y, idx_z, indexing="ij"
                                    )
                                    indices = zz + yy * nelz + xx * nely * nelz
                                elif not is_3d:
                                    xx, yy = np.meshgrid(idx_x, idx_y, indexing="ij")
                                    indices = yy + xx * nely

                        elif p["rshape"][i] == "◯":  # Circle/Sphere
                            if len(idx_x) > 0 and len(idx_y) > 0:
                                if is_3d and len(idx_z) > 0:
                                    i_grid, j_grid, k_grid = np.meshgrid(
                                        idx_x, idx_y, idx_z, indexing="ij"
                                    )
                                    mask = (i_grid - p["rx"][i]) ** 2 + (
                                        j_grid - p["ry"][i]
                                    ) ** 2 + (k_grid - p["rz"][i]) ** 2 <= p["rradius"][
                                        i
                                    ] ** 2
                                    ii, jj, kk = (
                                        i_grid[mask],
                                        j_grid[mask],
                                        k_grid[mask],
                                    )
                                    indices = kk + jj * nelz + ii * nely * nelz
                                elif not is_3d:
                                    i_grid, j_grid = np.meshgrid(
                                        idx_x, idx_y, indexing="ij"
                                    )
                                    mask = (i_grid - p["rx"][i]) ** 2 + (
                                        j_grid - p["ry"][i]
                                    ) ** 2 <= p["rradius"][i] ** 2
                                    ii, jj = i_grid[mask], j_grid[mask]
                                    indices = jj + ii * nely
                        self.xPhys[indices.flatten()] = (
                            1e-6 if p["rstate"][i] == "Void" else 1.0
                        )
                self.plot_material(ax, is_3d=is_3d)
            # Show initial message if xPhys is not a result (even partial) of optimization
            if self.footer.create_button.graphicsEffect() is not None:
                init_message = 'Configure parameters and press "Create"'
                if is_3d:
                    ax.text(
                        0.5,
                        0.5,
                        0.5,
                        s=init_message,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=16,
                        alpha=0.5,
                        color="black",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        s=init_message,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=16,
                        alpha=0.5,
                        color="black",
                    )

        self.redraw_non_material_layers(ax, is_3d)
        if not is_3d:
            ax.set_aspect("equal", "box")
        ax.autoscale(tight=True)
        self.canvas.draw()

    def plot_material(self, ax, is_3d, xPhys_data=None):
        """Plot the material."""
        p = self.last_params
        nelx, nely, nelz = p["nelxyz"]
        data_to_plot = self.xPhys if xPhys_data is None else xPhys_data
        if data_to_plot is None:
            return

        ax.clear()
        if is_3d:
            # Plot using voxels -> only the exterior box is visible
            # x_phys_3d = data_to_plot.reshape((nelz, nelx, nely)).transpose(1, 2, 0) if xPhys_data is None else xPhys_data.reshape((nelz, nelx, nely)).transpose(1, 2, 0)
            # base_color = np.array(to_rgb(self.material_widget.mat_color.get_color()))
            # color = np.zeros(x_phys_3d.shape + (4,))
            # color[..., :3] = base_color  # Set RGB
            # color[..., 3] = np.clip(x_phys_3d, 0.0, 1.0)
            # ax.voxels(x_phys_3d, facecolors=color, edgecolor=None, ) # Very slow for large grids
            # ax.set_box_aspect([nelx, nely, nelz])
            # self.redraw_non_material_layers(ax, is_3d_mode=True)

            p = self.last_params

            # Avoids plotting fully transparent points.
            visible_elements_mask = data_to_plot > 0.01
            visible_indices = np.where(visible_elements_mask)[0]

            densities = data_to_plot[visible_indices]

            z = visible_indices // (nelx * nely)
            x = (visible_indices % (nelx * nely)) // nely
            y = visible_indices % nely

            colors = np.zeros((len(densities), 4))
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            colors[:, :3] = base_color_rgb  # Set the RGB color for all points
            colors[:, 3] = densities  # Set the Alpha channel to the density

            ax.scatter(
                x + 0.5,
                y + 0.5,
                z + 0.5,
                s=6000 / max(nelx, nely, nelz),
                marker="s",  # Square markers to mimic voxels
                c=colors,
                alpha=None,
            )  # Alpha is now controlled by the 'c' array

            ax.set_box_aspect([nelx, nely, nelz])
        else:
            mat_color = self.material_widget.mat_color.get_color()
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", mat_color]
            )

            ax.imshow(
                data_to_plot.reshape((nelx, nely)).T,
                cmap=cmap,
                interpolation="nearest",
                origin="lower",
                norm=plt.Normalize(0, 1),
            )

    def redraw_non_material_layers(self, ax, is_3d):
        """Helper to draw all the plot layers that are NOT the main result."""
        # Layer 2: Overlays
        self.plot_forces(ax, is_3d=is_3d)
        self.plot_supports(ax, is_3d=is_3d)
        self.plot_regions(ax, is_3d=is_3d)
        self.plot_dimensions_frame(ax, is_3d=is_3d)
        self.plot_displacement_preview(ax, is_3d=is_3d)

    def plot_dimensions_frame(self, ax, is_3d):
        """Draws a dotted frame around the design space, controlled by the Dimensions section's visibility button."""
        if not self.sections["dimensions"].visibility_button.isChecked():
            ax.set_xlabel("")
            ax.set_ylabel("")
            if is_3d:
                ax.set_zlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if is_3d:
                ax.set_zticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return

        p = self.last_params
        nelx, nely, nelz = p["nelxyz"]

        if is_3d:
            # Define the 8 vertices of the box
            verts = [
                (0, 0, 0),
                (nelx, 0, 0),
                (nelx, nely, 0),
                (0, nely, 0),
                (0, 0, nelz),
                (nelx, 0, nelz),
                (nelx, nely, nelz),
                (0, nely, nelz),
            ]
            # Define the 12 edges by connecting the vertices
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for edge in edges:
                points = [verts[edge[0]], verts[edge[1]]]
                x, y, z = zip(*points)
                ax.plot(x, y, z, color="gray", linestyle=":", linewidth=1.5)
        else:
            rect = Rectangle(
                (0, 0),
                nelx,
                nely,
                fill=False,
                edgecolor="gray",
                linestyle=":",
                linewidth=1.5,
            )
            ax.add_patch(rect)

        ax.set_xlabel("X", color="black")
        ax.set_ylabel("Y", color="black")
        ax.yaxis.label.set_rotation(0)  # Display Y label vertically
        if is_3d:
            ax.set_zlabel("Z", color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        if is_3d:
            ax.tick_params(axis="z", colors="black")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")

    def plot_forces(self, ax, is_3d):
        """Plots the forces as arrows."""
        if not self.sections["forces"].visibility_button.isChecked():
            return
        if not self.last_params:
            return
        p = self.last_params
        if self.is_displaying_deformation and self.u is not None:
            if (
                self.displacement_widget.mov_iter.value() == 1
            ):  # Only show forces in single-frame displacement mode, not supported yet in animation mode
                for xkey, ykey, dirkey, col in [
                    ("fix", "fiy", "fidir", "r"),  # input forces
                    ("fox", "foy", "fodir", "b"),  # output forces
                ]:
                    active_forces = [
                        g
                        for g in self.forces_widget.inputs
                        if dirkey in g and g[dirkey].currentText() != "-"
                    ]
                    if len(active_forces) == 0:
                        continue

                    orig_fx = np.array([g[xkey].value() for g in active_forces])
                    orig_fy = np.array([g[ykey].value() for g in active_forces])
                    if is_3d:
                        if xkey == "fix":
                            orig_fz = np.array(
                                [g["fiz"].value() for g in active_forces]
                            )
                        else:
                            orig_fz = np.array(
                                [g["foz"].value() for g in active_forces]
                            )

                    colors = [
                        col
                        for g in self.forces_widget.inputs
                        if dirkey in g and g[dirkey].currentText() != "-"
                    ]
                    disp_factor = self.displacement_widget.mov_disp.value()
                    nely = p["nelxyz"][1]
                    indices = (
                        (orig_fz * (orig_fx + 1) * (nely + 1))
                        + (orig_fx * (nely + 1))
                        + orig_fy
                        if is_3d
                        else (orig_fx * (nely + 1)) + orig_fy
                    )

                    elemndof = 3 if is_3d else 2  # Degrees of freedom per element
                    ux = self.u[elemndof * indices, 0] * disp_factor
                    uy = self.u[elemndof * indices + 1, 0] * disp_factor
                    if is_3d:
                        uz = self.u[elemndof * indices + 2, 0] * disp_factor

                    new_fx = orig_fx + ux
                    new_fy = orig_fy + uy if is_3d else orig_fy - uy
                    if is_3d:
                        new_fz = orig_fz + uz

                    length = np.mean(p["nelxyz"][:2]) / 6
                    dx, dy = np.zeros_like(new_fx), np.zeros_like(new_fy)
                    if is_3d:
                        dz = np.zeros_like(new_fz)

                    directions = [g[dirkey].currentText() for g in active_forces]
                    for i, d in enumerate(directions):
                        if d == "-":
                            continue
                        if "X:→" in d:
                            dx[i] = length
                        elif "X:←" in d:
                            dx[i] = -length
                        elif "Y:↑" in d:
                            dy[i] = length
                        elif "Y:↓" in d:
                            dy[i] = -length
                        elif is_3d:
                            if "Z:<" in d:
                                dz[i] = length
                            elif "Z:>" in d:
                                dz[i] = -length

                    if is_3d:
                        ax.quiver(
                            new_fx,
                            new_fy,
                            new_fz,
                            dx,
                            dy,
                            dz,
                            color=colors,
                            length=length,
                            normalize=True,
                        )
                    else:
                        ax.quiver(
                            new_fx,
                            new_fy,
                            dx,
                            dy,
                            color=colors,
                            scale_units="xy",
                            angles="xy",
                            scale=1,
                        )
        else:
            for xkey, ykey, zkey, dirkey, col in [
                ("fix", "fiy", "fiz", "fidir", "r"),  # input forces
                ("fox", "foy", "foz", "fodir", "b"),  # output forces
            ]:
                directions = p[dirkey]
                if all(d == "-" for d in directions):
                    continue

                colors = [col for d in directions if d != "-"]
                dx, dy = np.zeros_like(p[xkey]), np.zeros_like(p[ykey])
                if is_3d:
                    dz = np.zeros_like(p[zkey])

                length = np.mean(p["nelxyz"][:2]) / 6

                for i, d in enumerate(directions):
                    if d == "-":
                        continue
                    if "X:→" in d:
                        dx[i] = length
                    elif "X:←" in d:
                        dx[i] = -length
                    elif "Y:↑" in d:
                        dy[i] = length
                    elif "Y:↓" in d:
                        dy[i] = -length
                    elif is_3d:
                        if "Z:<" in d:
                            dz[i] = length
                        elif "Z:>" in d:
                            dz[i] = -length

                if is_3d:
                    ax.quiver(
                        p[xkey],
                        p[ykey],
                        p[zkey],
                        dx,
                        dy,
                        dz,
                        color=colors,
                        length=length,
                        normalize=True,
                    )
                else:
                    ax.quiver(
                        p[xkey],
                        p[ykey],
                        dx,
                        dy,
                        color=colors,
                        scale_units="xy",
                        angles="xy",
                        scale=1,
                    )

    def plot_supports(self, ax, is_3d):
        """Plots the supports as triangles."""
        if not self.sections["supports"].visibility_button.isChecked():
            return
        # No need to consider the case is_displaying_deformation since the supports don't move
        p = self.last_params
        for i, d in enumerate(p["sdim"]):
            if d == "-":
                continue
            pos = [p["sx"][i], p["sy"][i], p["sz"][i]]
            if is_3d:
                ax.scatter(
                    pos[0],
                    pos[1],
                    pos[2],
                    s=80,
                    marker="^",
                    c="black",
                    depthshade=False,
                )
            else:
                ax.scatter(pos[0], pos[1], s=80, marker="^", c="black")

    def plot_regions(self, ax, is_3d):
        """Plots the regions outline (square/cube or circle/sphere) in 2D or 3D."""
        if not self.sections["regions"].visibility_button.isChecked():
            return
        if self.is_displaying_deformation:
            return  # Region are not relevant in deformation view

        p = self.last_params
        if not p:
            return
        for i, d in enumerate(p["rshape"]):
            shape = p["rshape"][i]
            if shape == "-":
                continue

            r = p["rradius"][i]
            rx, ry = p["rx"][i], p["ry"][i]

            if is_3d:
                rz = p["rz"][i]
                if shape == "□":  # Square/Cube
                    # Define the 8 vertices of the cube
                    verts = np.array(
                        [
                            [rx - r, ry - r, rz - r],
                            [rx + r, ry - r, rz - r],
                            [rx + r, ry + r, rz - r],
                            [rx - r, ry + r, rz - r],
                            [rx - r, ry - r, rz + r],
                            [rx + r, ry - r, rz + r],
                            [rx + r, ry + r, rz + r],
                            [rx - r, ry + r, rz + r],
                        ]
                    )
                    # Define the 12 edges connecting the vertices
                    edges = [
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        (3, 0),
                        (4, 5),
                        (5, 6),
                        (6, 7),
                        (7, 4),
                        (0, 4),
                        (1, 5),
                        (2, 6),
                        (3, 7),
                    ]
                    for edge in edges:
                        points = verts[list(edge)]
                        # Note: Matplotlib's 3D axes are ordered (X, Y, Z)
                        ax.plot(
                            points[:, 0],
                            points[:, 1],
                            points[:, 2],
                            color="green",
                            linestyle=":",
                        )

                elif shape == "◯":  # Circle/Sphere
                    # Create the surface grid for the sphere
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    # Parametric equations for a sphere
                    x = rx + r * np.outer(np.cos(u), np.sin(v))
                    y = ry + r * np.outer(np.sin(u), np.sin(v))
                    z = rz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_wireframe(x, y, z, color="green", linestyle=":")

            else:
                if shape == "□":  # Square/Cube
                    rect = plt.Rectangle(
                        (rx - r, ry - r),
                        2 * r,
                        2 * r,
                        fill=False,
                        edgecolor="green",
                        linestyle=":",
                    )
                    ax.add_patch(rect)
                elif shape == "◯":  # Circle/Sphere
                    circ = plt.Circle(
                        (rx, ry), r, fill=False, edgecolor="green", linestyle=":"
                    )
                    ax.add_patch(circ)

    def show_blank_plot(self):
        """Clears the canvas and displays a blank white plot."""
        self.figure.clear()
        self.figure.patch.set_facecolor("white")
        ax = self.figure.add_subplot(111, facecolor="white")

        # Ensure there are no ticks or labels
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide the black border frame

        self.canvas.draw()

    def plot_displacement_preview(self, ax, is_3d):
        """Overlays displacement vectors (quivers) on the plot if the preview is active."""
        if not self.sections["displacement"].visibility_button.isChecked():
            return
        if self.is_displaying_deformation:
            return  # The displacement vector doesn't match the deformed shape
        if self.u is None or self.xPhys is None:
            return

        p = self.last_params
        disp_factor = self.displacement_widget.mov_disp.value()
        factor = (
            disp_factor / np.mean(p["finorm"][0])
            if np.mean(p["finorm"][0]) != 0
            else disp_factor
        )

        nelx, nely, nelz = p["nelxyz"]
        if is_3d:
            step = max(
                1, int((nelx + nely + nelz) / 15)
            )  # number of elements to skip between 2 arrows
            x_coords, y_coords, z_coords = np.meshgrid(
                np.arange(0, nelx, step),
                np.arange(0, nely, step),
                np.arange(0, nelz, step),
                indexing="xy",
            )

            el_indices = (
                z_coords * (nelx * nely) + x_coords * nely + y_coords
            ).flatten()
            node_indices = (
                z_coords * ((nelx + 1) * (nely + 1)) + x_coords * (nely + 1) + y_coords
            ).flatten()
            material_mask = (
                self.xPhys[el_indices] > 0.5
            )  # Only show arrows in material regions

            # Get the coordinates and displacement vectors for the valid points
            x_valid = x_coords.flatten()[material_mask] + 0.5  # Center of element
            y_valid = y_coords.flatten()[material_mask] + 0.5
            z_valid = z_coords.flatten()[material_mask] + 0.5
            node_valid = node_indices[material_mask]

            ux = self.u[3 * node_valid, 0] * factor
            uy = -self.u[3 * node_valid + 1, 0] * factor
            uz = self.u[3 * node_valid + 2, 0] * factor

            ax.quiver(
                x_valid,
                y_valid,
                z_valid,
                ux,
                uy,
                uz,
                color="red",
                length=disp_factor / 4,
                normalize=True,
            )
        else:
            step = max(
                1, int((nelx + nely) / 25)
            )  # number of elements to skip between 2 arrows
            x_coords, y_coords = np.meshgrid(
                np.arange(0, nelx, step), np.arange(0, nely, step), indexing="xy"
            )

            el_indices = (x_coords * nely + y_coords).flatten()
            node_indices = (x_coords * (nely + 1) + y_coords).flatten()

            material_mask = (
                self.xPhys[el_indices] > 0.5
            )  # Only show arrows in material regions

            # Get the coordinates and displacement vectors for the valid points
            x_valid = x_coords.flatten()[material_mask]
            y_valid = y_coords.flatten()[material_mask]
            node_valid = node_indices[material_mask]

            ux = self.u[2 * node_valid, 0] * factor
            uy = -self.u[2 * node_valid + 1, 0] * factor

            ax.quiver(
                x_valid,
                y_valid,
                ux,
                uy,
                color="red",
                scale=40,
                scale_units="xy",
                angles="xy",
            )

    ###########
    # Presets #
    ###########

    def load_presets(self):
        """Loads presets from the JSON file and populates the combo box."""
        try:
            with open(self.presets_file, "r") as f:
                self.presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.presets = {}
            print("Presets file not found or invalid. Starting fresh.")

        self.preset.presets_combo.blockSignals(True)
        self.preset.presets_combo.clear()
        self.preset.presets_combo.addItem("Select a preset...")  # Index 0
        self.preset.presets_combo.addItems(sorted(self.presets.keys()))
        self.preset.presets_combo.blockSignals(False)
        self.preset.delete_preset_button.setEnabled(False)

    def save_presets(self):
        """Saves the current presets dictionary to the JSON file."""
        with open(self.presets_file, "w") as f:
            json.dump(self.presets, f, indent=4)
        self.status_bar.showMessage("Presets saved.", 3000)

    def save_new_preset(self):
        """Prompts the user for a name and saves the current parameters."""
        preset_name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter a name for this preset:"
        )

        if ok and preset_name:
            if preset_name in self.presets:
                reply = QMessageBox.question(
                    self,
                    "Overwrite Preset",
                    f"A preset named '{preset_name}' already exists. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled

            self.presets[preset_name] = self.gather_parameters()
            self.save_presets()
            self.load_presets()
            self.preset.presets_combo.setCurrentText(preset_name)
            self.preset.delete_preset_button.setEnabled(True)

    def delete_selected_preset(self):
        """Deletes the currently selected preset after confirmation."""
        preset_name = self.preset.presets_combo.currentText()
        if preset_name not in self.presets:
            return

        reply = QMessageBox.warning(
            self,
            "Delete Preset",
            f"You are about to permanently delete the preset '{preset_name}'.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.presets[preset_name]
            self.save_presets()
            self.load_presets()  # Reloads the list and disables the delete button
            self.status_bar.showMessage(f"Preset '{preset_name}' deleted.", 3000)

    def on_preset_selected(self):
        """Applies the parameters when a preset is selected from the combo box."""
        preset_name = self.preset.presets_combo.currentText()
        if preset_name in self.presets:
            self.apply_parameters(self.presets[preset_name])
            self.preset.delete_preset_button.setEnabled(True)
        else:
            self.preset.delete_preset_button.setEnabled(False)

    def apply_parameters(self, params):
        """Sets all UI widgets to the values from a given parameter dictionary."""
        self.block_all_parameter_signals(True)

        # Dimensions
        self.dim_widget.nx.setValue(params["nelxyz"][0])
        self.dim_widget.ny.setValue(params["nelxyz"][1])
        self.dim_widget.nz.setValue(params["nelxyz"][2])
        self.dim_widget.volfrac.setValue(params.get("volfrac", 0.3))
        self.update_position_ranges()

        # Regions
        for i, region_group in enumerate(self.regions_widget.inputs):
            region_group["rshape"].blockSignals(
                True
            )  # Reblock to avoid triggering on change
            region_group["rstate"].blockSignals(
                True
            )  # Reblock to avoid triggering on change
            region_group["rshape"].setCurrentText(params["rshape"][0])
            region_group["rstate"].setCurrentText(params["rstate"][0])
            region_group["rradius"].setValue(params["rradius"][0])
            region_group["rx"].setValue(params["rx"][0])
            region_group["ry"].setValue(params["ry"][0])
            region_group["rz"].setValue(params["rz"][0])

        # Forces
        nb_input_forces = 0
        for i, force_group in enumerate(self.forces_widget.inputs):
            if "fix" in force_group:
                force_group["fix"].setValue(params["fix"][i])
                force_group["fiy"].setValue(params["fiy"][i])
                force_group["fiz"].setValue(params["fiz"][i])
                force_group["fidir"].setCurrentText(params["fidir"][i])
                force_group["finorm"].setValue(params["finorm"][i])
                nb_input_forces += 1
            else:
                force_group["fox"].setValue(params["fox"][i - nb_input_forces])
                force_group["foy"].setValue(params["foy"][i - nb_input_forces])
                force_group["foz"].setValue(params["foz"][i - nb_input_forces])
                force_group["fodir"].setCurrentText(
                    params["fodir"][i - nb_input_forces]
                )
                force_group["fonorm"].setValue(params["fonorm"][i - nb_input_forces])

        # Supports
        num_supports_in_preset = len(params.get("sx", []))
        for i, support_group in enumerate(self.supports_widget.inputs):
            if i < num_supports_in_preset:
                # If data exists for this support, apply it
                support_group["sx"].setValue(params["sx"][i])
                support_group["sy"].setValue(params["sy"][i])
                support_group["sz"].setValue(params["sz"][i])
                support_group["sdim"].setCurrentText(params["sdim"][i])
            else:
                # If no data exists, reset this row to default empty values
                support_group["sx"].setValue(0)
                support_group["sy"].setValue(0)
                support_group["sz"].setValue(0)
                support_group["sdim"].setCurrentIndex(0)  # Set to '-'

        # Material
        self.material_widget.mat_E.setValue(params["E"])
        self.material_widget.mat_nu.setValue(params["nu"])
        self.material_widget.mat_init_type.setCurrentIndex(params["init_type"])

        # Optimizer
        self.optimizer_widget.opt_ft.setCurrentIndex(
            0 if params["filter_type"] == "Sensitivity" else 1
        )
        self.optimizer_widget.opt_fr.setValue(params["filter_radius_min"])
        self.optimizer_widget.opt_p.setValue(params["penal"])
        self.optimizer_widget.opt_max_change.setValue(params["max_change"])
        self.optimizer_widget.opt_n_it.setValue(params["n_it"])

        # Displacement
        self.displacement_widget.mov_disp.setValue(params["disp_factor"])
        self.displacement_widget.mov_iter.setValue(params["disp_iterations"])

        # Unblock signals
        self.block_all_parameter_signals(False)

        # Manually trigger a single update
        self.update_position_ranges()
        self.on_parameter_changed()
        self.status_bar.showMessage(
            f"Loaded preset: {self.preset.presets_combo.currentText()}", 3000
        )

    ############
    # BINARIZE #
    ############

    def on_binarize_clicked(self):
        """Applies a 0-or-1 threshold to the displayed result."""
        if self.xPhys is None:
            return

        # Define a threshold. Values below this are considered "white".
        threshold = 0.5

        # Use NumPy's fast vectorized 'where' function
        self.xPhys = np.where(self.xPhys > threshold, 1.0, 0.0)

        self.replot()
        self.status_bar.showMessage("View binarized.", 3000)

        self.footer.binarize_button.setEnabled(False)

    #########
    # THEME #
    #########

    def set_theme(self, theme_name, initial_setup=False):
        """Applies a theme stylesheet to the application GUI only."""
        stylesheet = (
            LIGHT_THEME_STYLESHEET if theme_name == "light" else DARK_THEME_STYLESHEET
        )
        QApplication.instance().setStyleSheet(stylesheet)
        self.current_theme = theme_name
        icons.set_theme(theme_name)

        # On the very first run, we need to style the plot once.
        # Afterwards, the plot theme is never touched again.
        if initial_setup:
            self.style_plot_default()

    def toggle_theme(self):
        """Switches between light and dark themes."""
        if self.current_theme == "light":
            self.set_theme("dark")
        else:
            self.set_theme("light")
        self.update_ui_icons()

    def update_ui_icons(self):
        """
        Resets all icons in the UI to force them to be re-fetched from the
        now theme-aware IconProvider.
        """
        # Update the theme button itself
        if self.current_theme == "dark":
            self.header.theme_button.setIcon(icons.get("sun"))
            self.header.theme_button.setToolTip("Switch to Light Theme")
        else:
            self.header.theme_button.setIcon(icons.get("moon"))
            self.header.theme_button.setToolTip("Switch to Dark Theme")

        # Update presets icons
        self.header.info_button.setIcon(icons.get("info"))
        self.preset.save_preset_button.setIcon(icons.get("save"))
        self.preset.delete_preset_button.setIcon(icons.get("delete"))
        # Update dimensions icons
        self.dim_widget.scale_button.setIcon(icons.get("scale"))
        # Update displacement icons
        self.displacement_widget.run_disp_button.setIcon(icons.get("move"))
        self.displacement_widget.stop_disp_button.setIcon(icons.get("stop"))
        self.displacement_widget.reset_disp_button.setIcon(icons.get("reset"))
        # Update footer icons
        self.footer.create_button.setIcon(icons.get("create"))
        self.footer.stop_button.setIcon(icons.get("Stop"))
        self.footer.save_button.setIcon(icons.get("save"))

        # Update dynamic icons (like the visibility and collapsible arrows)
        for section in self.sections.values():
            section.update_all_icons()

    def open_github_link(self):
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/Topopt-Comec")
        QDesktopServices.openUrl(url)
