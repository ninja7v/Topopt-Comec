# app/ui/main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Main window for the Topopt Comec application using PySide6.

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
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
    MaterialsWidget,
    OptimizerWidget,
    AnalysisWidget,
    PresetWidget,
    RegionsWidget,
    SupportWidget,
)
from .workers import AnalysisWorker, DisplacementWorker, OptimizerWorker
from .plotting import PlottingMixin
from .parameter_manager import ParameterManagerMixin


class MainWindow(QMainWindow, PlottingMixin, ParameterManagerMixin):
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
        self.sections["Dimensions"] = self.create_Dimensions_section()
        self.sections["Regions"] = self.create_regions_section()
        self.sections["Forces"] = self.create_forces_section()
        self.sections["Supports"] = self.create_supports_section()
        self.sections["Materials"] = self.create_materials_section()
        self.sections["Optimizer"] = self.create_optimizer_section()
        self.sections["Analysis"] = self.create_analysis_section()
        self.sections["Displacement"] = self.create_displacement_section()

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

    def create_Dimensions_section(self):
        """Creates the first section for Dimensions and volume fraction."""
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
        self.regions_widget.add_btn.clicked.connect(self.connect_region_signals)
        self.regions_widget.nbRegionsChanged.connect(self.on_parameter_changed)
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
        self.supports_widget.add_btn.clicked.connect(self.connect_support_signals)
        self.supports_widget.add_btn.clicked.connect(self.update_position_ranges)
        self.supports_widget.nbSupportsChanged.connect(self.on_parameter_changed)

        return section

    def create_materials_section(self):
        """Creates the fifth section for material properties."""
        self.materials_widget = MaterialsWidget()
        section = CollapsibleSection("🧱 Material", self.materials_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        self.materials_widget.add_btn.clicked.connect(self.connect_material_signals)
        self.materials_widget.nbMaterialsChanged.connect(self.on_parameter_changed)
        self.materials_widget.mat_init_type.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        return section

    def connect_material_signals(self):
        """(Re)connects on_parameter_changed signals to all current material widgets."""
        for mw in self.materials_widget.inputs:
            mw["color"].clicked.connect(self.replot)
            mw["E"].valueChanged.connect(self.on_parameter_changed)
            mw["nu"].valueChanged.connect(self.on_parameter_changed)
            mw["percent"].valueChanged.connect(self.on_parameter_changed)

    def create_optimizer_section(self):
        """Creates the sixth section for optimization parameters."""
        self.optimizer_widget = OptimizerWidget()
        section = CollapsibleSection("💻 Optimizer", self.optimizer_widget)
        self.optimizer_widget.opt_ft.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_fr.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_p.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_eta.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_max_change.valueChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_n_it.valueChanged.connect(self.on_parameter_changed)
        return section

    def create_analysis_section(self):
        """Creates the seventh section for analysis parameters."""
        self.analysis_widget = AnalysisWidget()
        section = CollapsibleSection("🔍 Analysis", self.analysis_widget)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        self.analysis_widget.stop_analysis_button.clicked.connect(self.stop_analysis)
        self.analysis_widget.run_analysis_button.setEnabled(
            False
        )  # Disabled until a result is ready
        self.analysis_widget.run_analysis_button.clicked.connect(self.run_analysis)
        self.analysis_widget.stop_analysis_button.clicked.connect(self.stop_analysis)
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
        footer_widget.save_3mf_action.triggered.connect(self.save_as_3mf)
        return footer_widget

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
        self.footer.create_button.setEnabled(False)
        self.footer.create_button.hide()
        self.footer.stop_button.setText(" Stop")
        self.footer.stop_button.setEnabled(True)
        self.footer.stop_button.show()
        self.footer.binarize_button.setEnabled(False)
        self.footer.save_button.setEnabled(False)
        self.status_bar.showMessage("Starting optimization...")
        self.progress_bar.setRange(0, self.last_params["Optimizer"]["n_it"])
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
        is_3d = (
            self.last_params["Dimensions"]["nelxyz"][2] > 0
            if self.last_params
            else False
        )

        # Multi-material frames have shape (n_mat, nel) — use full replot
        is_multi = xPhys_frame.ndim == 2

        if is_3d or is_multi:
            self.plot_material(ax, is_3d=is_3d, xPhys_data=xPhys_frame)
            self.redraw_non_material_layers(ax, is_3d=is_3d)
        else:
            if not ax.images:
                return
            im = ax.images[0]  # The imshow object is the first image on the axes
            nelx, nely = self.last_params["Dimensions"]["nelxyz"][:2]
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
        self.analysis_widget.run_analysis_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.sections["Displacement"].visibility_button.setEnabled(True)
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
        if self.last_params["Displacement"]["disp_iterations"] == 1:
            # Run single-frame logic directly
            self.status_bar.showMessage("Calculating single displacement frame...")
            QApplication.processEvents()  # Update UI
            from app.core.displacements import single_linear_displacement

            nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
            self.last_displayed_frame_data = single_linear_displacement(
                self.u,
                nelx,
                nely,
                nelz,
                self.last_params["Displacement"]["disp_factor"],
            )
            self.replot()
            self.handle_displacement_finished("Single frame shown.")
            self.status_bar.showMessage("Single displacement plot shown.", 3000)
        else:
            self.footer.create_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.status_bar.showMessage("Starting displacement computation...")

            self.progress_bar.setRange(
                0, self.last_params["Displacement"]["disp_iterations"] + 1
            )
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
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
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
            base_color_rgb = to_rgb(
                self.materials_widget.inputs[0]["color"].get_color()
            )
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
        if self.sections["Displacement"].visibility_button.isChecked():
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

    ############
    # Analysis #
    ############
    def run_analysis(self):
        """Starts the analysis evaluation based on the last optimization result"""
        if self.xPhys is None or self.u is None:
            QMessageBox.warning(
                self,
                "Analysis Error",
                "You must run a successful optimization before analyzing results.",
            )
            return

        aw = self.analysis_widget
        aw.run_analysis_button.setEnabled(False)
        aw.stop_analysis_button.setText(" Stop")
        aw.button_stack.setCurrentWidget(aw.stop_analysis_button)
        aw.stop_analysis_button.setEnabled(True)
        self.footer.create_button.setEnabled(False)
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.displacement_widget.run_disp_button.setEnabled(False)
        self.footer.create_button.setEnabled(False)

        self.analysis_worker = AnalysisWorker(self.last_params, self.xPhys, self.u)
        self.analysis_worker.progress.connect(self.update_analysis_progress)
        self.analysis_worker.finished.connect(self.handle_analysis_finished)
        self.analysis_worker.error.connect(self.handle_analysis_error)
        self.analysis_worker.start()

    def stop_analysis(self):
        """Requests the running analysis worker to stop."""
        if self.analysis_worker:
            self.analysis_widget.stop_analysis_button.setText(" Stopping...")
            self.analysis_widget.stop_analysis_button.setEnabled(False)
            self.analysis_worker.request_stop()

    def update_analysis_progress(self, iteration):
        """Updates the progress bar and status message during analysis."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(f"Running analysis: step {iteration}...")

    def handle_analysis_finished(self, results):
        """Handles the results after analysis finishes successfully."""
        aw = self.analysis_widget
        aw.checkerboard_result.setText("yes" if results[0] else "no")
        aw.checkerboard_result.setStyleSheet(
            "color: green;" if not results[0] else "color: red;"
        )
        aw.watertight_result.setText("yes" if results[1] else "no")
        aw.watertight_result.setStyleSheet(
            "color: green;" if results[1] else "color: red;"
        )
        aw.threshold_result.setText("yes" if results[2] else "no")
        aw.threshold_result.setStyleSheet(
            "color: green;" if results[2] else "color: red;"
        )
        aw.efficiency_result.setText("yes" if results[3] else "no")
        aw.efficiency_result.setStyleSheet(
            "color: green;" if results[3] else "color: red;"
        )
        self.status_bar.showMessage("Analysis finished successfully.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        aw.stop_analysis_button.setEnabled(False)
        aw.stop_analysis_button.setText(" Stop")  # Reset text for next run
        aw.button_stack.setCurrentWidget(aw.run_analysis_button)
        aw.run_analysis_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.footer.create_button.setEnabled(True)

    def handle_analysis_error(self, error_msg):
        """Handles any errors that occur during analysis."""
        self.status_bar.showMessage("Analysis failed.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.analysis_widget.run_analysis_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Runtime Error", error_msg)

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
                "result_3d"
                if self.last_params["Dimensions"]["nelxyz"][2] > 0
                else "result_2d"
            )

        # File dialog config
        filters = {
            "png": ("Save as PNG", "Portable Network Graphics (*.png)"),
            "vti": ("Save as VTI", "VTK Image Data (*.vti)"),
            "stl": ("Save as STL", "STL File (*.stl)"),
            "3mf": ("Save as 3MF", "3D Manufacturing File (*.3mf)"),
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
                    self.xPhys, self.last_params["Dimensions"]["nelxyz"], filepath
                )
                if not success:
                    raise Exception(error_msg)

            elif file_type == "stl":
                success, error_msg = exporters.save_as_stl(
                    self.xPhys, self.last_params["Dimensions"]["nelxyz"], filepath
                )
                if not success:
                    raise Exception(error_msg)

            elif file_type == "3mf":
                success, error_msg = exporters.save_as_3mf(
                    self.xPhys, self.last_params["Dimensions"]["nelxyz"], filepath
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

    def save_as_3mf(self):
        """Function connected to the save as 3MF button."""
        self.save_result_as("3mf")

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

    def _plot_deformation(self, ax, is_3d, nelx, nely, nelz):
        if (
            self.last_params["Displacement"]["disp_iterations"] == 1
        ):  # Single-frame grid plot
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
                colors[:, :3] = to_rgb(
                    self.materials_widget.inputs[0]["color"].get_color()
                )
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
                hex_color = to_hex(self.materials_widget.inputs[0]["color"].get_color())
                color_cmap = LinearSegmentedColormap.from_list(
                    "material_shades",
                    [hex_color, "#ffffff"],  # selected material color → white
                )
                X, Y = self.last_displayed_frame_data
                ax.pcolormesh(
                    X,
                    Y,
                    -self.xPhys.reshape((nelx, nely)),
                    cmap=color_cmap,
                    shading="auto",
                )
        # Multi-iteration displacement handled in update_animation_frame

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

        # --- Dimensions ---
        Dimensions = params.get("Dimensions", {})
        self.dim_widget.nx.setValue(Dimensions.get("nelxyz", [1, 1, 1])[0])
        self.dim_widget.ny.setValue(Dimensions.get("nelxyz", [1, 1, 1])[1])
        self.dim_widget.nz.setValue(Dimensions.get("nelxyz", [1, 1, 1])[2])
        self.dim_widget.volfrac.setValue(Dimensions.get("volfrac", 0.3))
        self.update_position_ranges()

        # --- Regions (optional) ---
        pr = params.get("Regions", {})
        num_regions_in_preset = len(pr.get("rshape", []))
        current_num_regions = len(self.regions_widget.inputs)

        while current_num_regions > num_regions_in_preset:
            self.regions_widget.remove_region(current_num_regions - 1, False)
            current_num_regions -= 1
        while current_num_regions < num_regions_in_preset:
            self.regions_widget.add_region(emit_signal=False)
            current_num_regions += 1

        for i in range(num_regions_in_preset):
            rw = self.regions_widget.inputs[i]
            rw["rshape"].blockSignals(True)
            rw["rstate"].blockSignals(True)
            rw["rshape"].setCurrentText(pr["rshape"][i])
            rw["rstate"].setCurrentText(pr["rstate"][i])
            rw["rradius"].setValue(pr["rradius"][i])
            rw["rx"].setValue(pr["rx"][i])
            rw["ry"].setValue(pr["ry"][i])
            rw["rz"].setValue(pr["rz"][i])
        self.connect_region_signals()

        # --- Forces ---
        pf = params.get("Forces", {})
        nb_input_forces = 0
        for i, fw in enumerate(self.forces_widget.inputs):
            if "fix" in fw:
                fw["fix"].setValue(pf["fix"][i])
                fw["fiy"].setValue(pf["fiy"][i])
                fw["fiz"].setValue(pf["fiz"][i])
                fw["fidir"].setCurrentText(pf["fidir"][i])
                fw["finorm"].setValue(pf["finorm"][i])
                nb_input_forces += 1
            else:
                fw["fox"].setValue(pf["fox"][i - nb_input_forces])
                fw["foy"].setValue(pf["foy"][i - nb_input_forces])
                fw["foz"].setValue(pf["foz"][i - nb_input_forces])
                fw["fodir"].setCurrentText(pf["fodir"][i - nb_input_forces])
                fw["fonorm"].setValue(pf["fonorm"][i - nb_input_forces])

        # --- Supports (optional) ---
        ps = params.get("Supports", {})
        num_supports_in_preset = len(ps.get("sx", []))
        current_num = len(self.supports_widget.inputs)
        while current_num > num_supports_in_preset:
            last_btn = self.supports_widget.inputs[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect(self.on_parameter_changed)
            except TypeError:
                pass
            self.supports_widget.remove_support(current_num - 1, False)
            current_num -= 1
        while current_num < num_supports_in_preset:
            self.supports_widget.add_support(None, "XYZ", False)
            current_num += 1
        for i in range(num_supports_in_preset):
            sw = self.supports_widget.inputs[i]
            sw["sx"].setValue(ps["sx"][i])
            sw["sy"].setValue(ps["sy"][i])
            sw["sz"].setValue(ps["sz"][i])
            sw["sdim"].setCurrentText(ps["sdim"][i])
            sw["sr"].setValue(ps["sr"][i])
        self.connect_support_signals()

        # --- Materials ---
        pm = params.get("Materials", {})
        num_material_in_preset = len(pm.get("E", []))
        current_num = len(self.materials_widget.inputs)
        while current_num > num_material_in_preset:
            last_btn = self.materials_widget.inputs[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect(self.on_parameter_changed)
            except TypeError:
                pass
            self.materials_widget.remove_material(current_num - 1, False)
            current_num -= 1
        while current_num < num_material_in_preset:
            self.materials_widget.add_material()
            current_num += 1
        for i in range(num_material_in_preset):
            mw = self.materials_widget.inputs[i]
            mw["E"].setValue(pm["E"][i])
            mw["nu"].setValue(pm["nu"][i])
            # percent and color can be optional, skip if missing
            percents = pm.get(
                "percent", [int(100 / num_material_in_preset)] * num_material_in_preset
            )
            mw["percent"].setValue(percents[i])
            color = pm.get("color", ["#000000"] * num_material_in_preset)
            mw["color"].set_color(color[i])
        self.materials_widget.mat_init_type.setCurrentIndex(pm.get("init_type", 0))
        self.connect_material_signals()

        # --- Optimizer ---
        po = params.get("Optimizer", {})
        self.optimizer_widget.opt_ft.setCurrentIndex(
            0
            if po.get("filter_type", "Sensitivity") == "Sensitivity"
            else 1 if po.get("filter_type", "Sensitivity") == "Density" else 2
        )
        self.optimizer_widget.opt_fr.setValue(po.get("filter_radius_min", 1.3))
        self.optimizer_widget.opt_p.setValue(po.get("penal", 3.0))
        self.optimizer_widget.opt_eta.setValue(po.get("eta", 0.3))
        self.optimizer_widget.opt_max_change.setValue(po.get("max_change", 0.05))
        self.optimizer_widget.opt_n_it.setValue(po.get("n_it", 30))
        self.optimizer_widget.opt_solver.setCurrentText(po.get("solver", "Auto"))

        # --- Displacement ---
        Displacement = params.get("Displacement", {})
        self.displacement_widget.mov_disp.setValue(Displacement.get("disp_factor", 1.0))
        self.displacement_widget.mov_iter.setValue(
            Displacement.get("disp_iterations", 1)
        )

        # --- Unblock signals ---
        self.block_all_parameter_signals(False)

        # Manually trigger a single update
        self.update_position_ranges()
        self.on_parameter_changed()
        self.status_bar.showMessage(
            f"Loaded preset: {self.preset.presets_combo.currentText()}", 3000
        )

    def connect_support_signals(self):
        """(Re)connects on_parameter_changed signals to all current support widgets."""
        for support_group in self.supports_widget.inputs:
            support_group["sx"].valueChanged.connect(self.on_parameter_changed)
            support_group["sy"].valueChanged.connect(self.on_parameter_changed)
            support_group["sz"].valueChanged.connect(self.on_parameter_changed)
            support_group["sdim"].currentIndexChanged.connect(self.on_parameter_changed)
            support_group["sr"].valueChanged.connect(self.on_parameter_changed)
            # there is a special "nbSupportsChanged" signal for the remove button

    def connect_region_signals(self):
        """(Re)connects on_parameter_changed signals to all current region widgets."""
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
            # there is a special "nbRegionsChanged" signal for the remove button

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
