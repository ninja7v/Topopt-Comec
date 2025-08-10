# app/ui/main_window.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Main window for the Topopt Comec application using PySide6.

import sys
import os
import json
import copy
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QSplitter, QScrollArea, QFrame,
                               QProgressBar, QMessageBox, QApplication, QInputDialog, QFileDialog)
from PySide6.QtGui import QDesktopServices

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
#import mcubes

from app.ui import exporters
from .worker import OptimizerWorker, DisplacementWorker
from .widgets import (HeaderWidget, PresetWidget,
                      CollapsibleSection, DimensionsWidget, VoidWidget,
                      ForcesWidget, SupportWidget, MaterialWidget,
                      OptimizerWidget, DisplacementWidget, FooterWidget)
from .icons import icons
from .themes import LIGHT_THEME_STYLESHEET, DARK_THEME_STYLESHEET

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Topopt Comec - Topology Optimization for Compliant Mechanisms")
        self.setGeometry(100, 100, 1280, 720)
        
        # Consolidate duplicate variable declarations
        self.xPhys = None
        self.u = None
        self.last_params = {}
        self.current_theme = 'dark'
        self.displacement_worker = None
        self.worker = None # To hold the optimizer worker

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
            self.on_preset_selected() # This applies the preset and replots
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
        self.sections['dimensions'] = self.create_dimensions_section()
        self.sections['void'] = self.create_void_section()
        self.sections['forces'] = self.create_forces_section()
        self.sections['supports'] = self.create_supports_section()
        self.sections['material'] = self.create_material_section()
        self.sections['optimizer'] = self.create_optimizer_section()
        self.sections['displacement'] = self.create_displacement_section()
        
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
        self.dim_widget.nx.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.ny.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self.on_mode_changed)
        return section

    def create_void_section(self):
        """Creates the second section for void region parameters."""
        self.void_widget = VoidWidget()
        section = CollapsibleSection("⚫ Void Region", self.void_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        self.void_widget.v_shape.currentIndexChanged.connect(self.on_parameter_changed)
        self.void_widget.v_radius.valueChanged.connect(self.on_parameter_changed)
        self.void_widget.v_cx.valueChanged.connect(self.on_parameter_changed)
        self.void_widget.v_cy.valueChanged.connect(self.on_parameter_changed)
        self.void_widget.v_cz.valueChanged.connect(self.on_parameter_changed)
        return section

    def create_forces_section(self):
        """Creates the third section for forces parameters."""
        self.forces_widget = ForcesWidget()
        section = CollapsibleSection("💪 Forces", self.forces_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)

        # 3. Connect the signals from the widgets INSIDE the ForcesWidget
        for force_group in self.forces_widget.inputs:
            force_group['fx'].valueChanged.connect(self.on_parameter_changed)
            force_group['fy'].valueChanged.connect(self.on_parameter_changed)
            force_group['fz'].valueChanged.connect(self.on_parameter_changed)
            force_group['a'].currentIndexChanged.connect(self.on_parameter_changed)
            force_group['fv'].valueChanged.connect(self.on_parameter_changed)
            
        return section

    def create_supports_section(self):
        """Creates the fourth section for supports parameters."""
        self.supports_widget = SupportWidget()
        section = CollapsibleSection("🔺 Supports", self.supports_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)

        for support_input_group in self.supports_widget.inputs:
            support_input_group['sx'].valueChanged.connect(self.on_parameter_changed)
            support_input_group['sy'].valueChanged.connect(self.on_parameter_changed)
            support_input_group['sz'].valueChanged.connect(self.on_parameter_changed)
            support_input_group['d'].currentIndexChanged.connect(self.on_parameter_changed)
            
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
        return section

    def create_optimizer_section(self):
        """Creates the sixth section for optimization parameters."""
        self.optimizer_widget = OptimizerWidget()
        section = CollapsibleSection("💻 Optimizer", self.optimizer_widget)
        self.optimizer_widget.opt_ft.currentIndexChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_fr.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_p.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_n_it.valueChanged.connect(self.on_parameter_changed)
        return section
    
    def create_displacement_section(self):
        """Creates the seventh section for displacement animation parameters."""
        self.displacement_widget = DisplacementWidget()
        section = CollapsibleSection("↔️ Displacement", self.displacement_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        section.visibility_button.setEnabled(False) # Disabled until a result is ready
        section.visibility_button.setToolTip("Preview displacement vectors on the main plot")
        section.visibility_button.setChecked(False)
        
        section.visibility_button.toggled.connect(self.replot)
        self.displacement_widget.run_disp_button.clicked.connect(self.run_displacement)
        self.displacement_widget.mov_disp.valueChanged.connect(self.on_displacement_preview_changed)
        
        return section
    
    def on_visibility_toggled(self, checked):
        """Handles the toggling of any visibility button."""
        # The 'sender()' method gives us the specific button that was clicked.
        button = self.sender()
        if not button:
            return

        if checked:
            button.setIcon(icons.get('eye_open'))
            button.setToolTip("Element is visible. Click to hide.")
        else:
            button.setIcon(icons.get('eye_closed'))
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
        nx, ny, nz = self.dim_widget.nx.value(), self.dim_widget.ny.value(), self.dim_widget.nz.value()
        params['nelxyz'] = [nx, ny, nz]
        params['volfrac'] = self.dim_widget.volfrac.value()

        # Void
        params['v'] = self.void_widget.v_shape.currentText()[0]
        params['r'] = self.void_widget.v_radius.value()
        params['c'] = [self.void_widget.v_cx.value(), self.void_widget.v_cy.value(), self.void_widget.v_cz.value()]
        
        # Forces
        params['fx'], params['fy'], params['fz'] = [], [], []
        params['a'], params['fv'] = [], []
        for fw in self.forces_widget.inputs:
            params['fx'].append(fw['fx'].value())
            params['fy'].append(fw['fy'].value())
            params['fz'].append(fw['fz'].value())
            params['a'].append(fw['a'].currentText())
            params['fv'].append(fw['fv'].value())
            
        # Supports
        params['sx'], params['sy'], params['sz'] = [], [], []
        params['dim'] = []
        for sw in self.supports_widget.inputs:
            params['sx'].append(sw['sx'].value())
            params['sy'].append(sw['sy'].value())
            params['sz'].append(sw['sz'].value())
            params['dim'].append(sw['d'].currentText())
            
        # Material
        params['E'] = self.material_widget.mat_E.value()
        params['nu'] = self.material_widget.mat_nu.value()
        
        # Optimizer
        params['ft'] = 'Sensitivity' if self.optimizer_widget.opt_ft.currentIndex() == 0 else 'Density'
        params['rmin'] = self.optimizer_widget.opt_fr.value()
        params['penal'] = self.optimizer_widget.opt_p.value()
        params['n_it'] = self.optimizer_widget.opt_n_it.value()

        # Movement
        params['disp_factor'] = self.displacement_widget.mov_disp.value()
        params['disp_iterations'] = self.displacement_widget.mov_iter.value()
        
        return params

    def on_parameter_changed(self):
        """React when a parameter is changed."""#
        # Play the animation
        self.footer.start_create_button_effect()
        
        # First, check if a valid result from a previous run exists.
        if self.xPhys is not None:
            self.xPhys = None
            self.u = None

            # Disable buttons that require a valid result
            self.footer.binarize_button.setEnabled(False)
            self.footer.save_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.sections['displacement'].visibility_button.setEnabled(False)
            
            # Inform the user what happened
            self.status_bar.showMessage("Parameters changed. Please run 'Create' for a new result.", 3000)
        
        self.last_params = self.gather_parameters()
        self.replot()
        
        # Check if the current state matches the selected preset
        current_preset_name = self.preset.presets_combo.currentText()
        if current_preset_name in self.presets:
            if not self.are_parameters_equivalent(self.presets[current_preset_name], self.last_params):
                # The parameters have changed, so deselect the preset
                self.preset.presets_combo.blockSignals(True)
                self.preset.presets_combo.setCurrentIndex(0) # Set to "Select a preset..."
                self.preset.presets_combo.blockSignals(False)
                self.preset.delete_preset_button.setEnabled(False)
    
    def are_parameters_equivalent(self, params1, params2):
        """
        Intelligently compares two parameter dictionaries, ignoring irrelevant data
        like inactive supports, forces, or void parameters.
        """
        # Create deep copies to avoid modifying the original dictionaries
        p1_norm = copy.deepcopy(params1)
        p2_norm = copy.deepcopy(params2)
        
        def normalize_params(p):
            if 'nelxyz' in p:
                is_2d = len(p['nelxyz']) < 3 or p['nelxyz'][2] == 0.0
                if is_2d:
                    p['nelxyz'] = p['nelxyz'][:2]
            # --- Normalize for 2D vs 3D ---
            if is_2d:
                # If it's a 2D problem, remove all Z-related keys for a clean comparison
                p['nelxyz'] = p['nelxyz'][:2] # Keep only Nx, Ny
                
                keys_to_clean = ['fz', 'sz', 'c']
                for key in keys_to_clean:
                    if key in p:
                        if isinstance(p[key], list) and len(p[key]) > 0 and isinstance(p[key][0], (int, float)):
                            # This is for lists of coordinates like c=[cx, cy, cz]
                            p[key] = p[key][:2]
                        elif isinstance(p[key], list) and len(p[key]) > 0 and isinstance(p[key][0], list):
                                # This is for lists of lists like sx=[[s1x], [s2x], ...] (not our current structure, but robust)
                            for i in range(len(p[key])):
                                p[key][i] = p[key][i][:2]
                
                # Specifically handle list of values like fz=[f1z, f2z, f3z]
                if 'fz' in p: p.pop('fz', None)
                if 'sz' in p: p.pop('sz', None)
                if 'c' in p: p['c'] = p['c'][:2]
            # --- Normalize Supports ---
            # Zip all support lists together, filter for active ones, then unzip.
            zipped_supports = zip(p.get('sx', []), p.get('sy', []), p.get('sz', []) if not is_2d else [0]*len(p.get('sx', [])), p.get('dim', []))
            active_supports = [s for s in zipped_supports if s[3] != '-']
            if active_supports:
                sx, sy, sz, dim = list(zip(*active_supports))
                p['sx'], p['sy'], p['sz'], p['dim'] = list(sx), list(sy), list(sz), list(dim)
            else:
                p['sx'], p['sy'], p['sz'], p['dim'] = [], [], [], []
            if is_2d and 'sz' in p:
                p.pop('sz')
            # --- Normalize Forces ---
            # The input force (index 0) is always kept.
            # We only filter the output forces (indices 1 and beyond).
            if 'fx' in p and 'a' in p:
                zipped_forces = zip(p.get('fx', []), p.get('fy', []), p.get('fz', []) if not is_2d else [0]*len(p.get('fx', [])), p.get('a', []), p.get('fv', []))
                force_list = list(zipped_forces)
                # Keep the input force, and any output forces that are active.
                active_forces = [force_list[0]] + [f for f in force_list[1:] if f[3] != '-']
                if active_forces:
                    fx, fy, fz, a, fv = list(zip(*active_forces))
                    p['fx'], p['fy'], p['fz'], p['a'], p['fv'] = list(fx), list(fy), list(fz), list(a), list(fv)
                else: # Should not happen as input force is required, but safe to have
                    p['fx'], p['fy'], p['fz'], p['a'], p['fv'] = [], [], [], [], []
            if is_2d and 'fz' in p:
                p.pop('fz')
            # --- Normalize Void ---
            # If the void shape is '-', the other parameters are meaningless.
            if 'v' in p:
                if p['v'] == '-':
                    p.pop('v', None)
                    p.pop('r', None)
                    p.pop('c', None)
            
            return p
        
        # Normalize both dictionaries
        p1_norm = normalize_params(p1_norm)
        p2_norm = normalize_params(p2_norm)

        # Now, compare the normalized, canonical versions
        return json.dumps(p1_norm, sort_keys=True) == json.dumps(p2_norm, sort_keys=True)
        
    def validate_parameters(self, p):
        """Checks for common input errors."""
        nx, ny, nz = p['nelxyz']
        if nx <= 0 or ny <= 0 or nz < 0: return "Nx, Ny, Nz must be positive."
        if p['a'][0] == '-': return "Input force (Force 1) direction must be set."
        if len(p['a']) > 2 and p['a'][1] == '-' and p['a'][2] == '-': 
            return "At least one output force must be set."
        elif len(p['a']) == 2 and p['a'][1] == '-':
            return "At least one output force must be set."
        elif len(p['a']) == 1:
            return "At least one output force must be set."
        has_support = any(d != '-' for d in p['dim'])
        if not has_support: return "At least one support must be defined."
        return None
    
    def update_position_ranges(self):
        """
        Updates the maximum values for all position-related spin boxes
        based on the current Nx, Ny, and Nz values.
        """
        # Get the current maximums from the dimensions widget
        # The check for self.dim_widget handles the initial app startup
        if not hasattr(self, 'dim_widget'):
            return
        
        # Get the current maximums from the dimensions widget
        nx = self.dim_widget.nx.value()
        ny = self.dim_widget.ny.value()
        nz = self.dim_widget.nz.value()

        # Update ranges for Void Widget
        self.void_widget.v_cx.setMaximum(nx)
        self.void_widget.v_cy.setMaximum(ny)
        self.void_widget.v_cz.setMaximum(nz)

        # Update ranges for all Forces
        for force_group in self.forces_widget.inputs:
            force_group['fx'].setMaximum(nx)
            force_group['fy'].setMaximum(ny)
            force_group['fz'].setMaximum(nz)

        # Update ranges for all Supports
        for support_group in self.supports_widget.inputs:
            support_group['sx'].setMaximum(nx)
            support_group['sy'].setMaximum(ny)
            support_group['sz'].setMaximum(nz)
    
    def on_mode_changed(self):
        """
        Called whenever the Nz value changes.
        Updates UI elements that depend on whether the mode is 2D or 3D.
        """
        is_3d_mode = self.dim_widget.nz.value() > 0
        
        # Tell the VoidWidget to update its labels
        if hasattr(self, 'void_widget'): # Check if it exists yet
            self.void_widget.update_for_mode(is_3d_mode)

    ################
    # OPTIMIZATION #
    ################

    def run_optimization(self):
        """Starts the optimization process based on current parameters, and gives live updates."""
        self.last_params = self.gather_parameters()
        error = self.validate_parameters(self.last_params)
        if error:
            QMessageBox.critical(self, "Input Error", error)
            return
        
        # Stop animation
        self.footer.stop_create_button_effect()
        
        # This creates the gray box that will be updated live
        self.replot()
        QApplication.processEvents() # Force the UI to draw the initial state
        
        self.preset.setEnabled(False)
        self.footer.create_button.hide()
        self.footer.stop_button.setText(" Stop")
        self.footer.stop_button.setEnabled(True)
        self.footer.stop_button.show()
        self.footer.create_button.setEnabled(False)
        self.footer.binarize_button.setEnabled(False)
        self.footer.save_button.setEnabled(False)
        self.status_bar.showMessage("Starting optimization...")
        self.progress_bar.setRange(0, self.last_params['n_it'])
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        for section in self.sections.values():
            section.collapse()

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
            self.footer.stop_button.setEnabled(False) # Prevent multiple clicks
            self.worker.request_stop()

    def update_optimization_progress(self, iteration, objective, change):
        """Updates the progress bar and status message during optimization."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(f"It: {iteration}, Obj: {objective:.4f}, Change: {change:.4f}")
        
    def update_optimization_plot(self, xPhys_frame):
        """Updates the plot with an intermediate frame from the optimizer."""
        # Ensure a plot exist to update
        if not self.figure.get_axes():
            return

        ax = self.figure.get_axes()[0]

        # Get dimensions and update the image data
        is_3d_mode = self.last_params['nelxyz'][2] > 0 if self.last_params else False
        if is_3d_mode:
            self.plot_material(ax, is_3d = is_3d_mode, xPhys_data=xPhys_frame)
            self.redraw_non_material_layers(ax, is_3d_mode=True)
        else:
            if not ax.images: return
            im = ax.images[0] # The imshow object is the first image on the axes
            nx, ny = self.last_params['nelxyz'][:2]
            im.set_array(xPhys_frame.reshape((nx, ny)).T)
        
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
        self.sections['displacement'].visibility_button.setEnabled(True)
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
        if self.xPhys is None:
            QMessageBox.warning(self, "Displacement Error", "You must run a successful optimization before analyzing movement.")
            return
        
        self.replot()
        QApplication.processEvents()

        params = self.gather_parameters()
        is_3d_mode = params['nelxyz'][2] > 0
        
        if params['disp_iterations'] == 1:
            # Run single-frame logic directly
            self.status_bar.showMessage("Calculating single displacement frame...")
            QApplication.processEvents() # Update UI
            
            if is_3d_mode:
                from app.displacements.displacement_3d import single_linear_displacement_3d
                plot_data = single_linear_displacement_3d(
                    self.xPhys, self.u, *params['nelxyz'], params['disp_factor']
                )
            else:
                from app.displacements.displacement_2d import single_linear_displacement_2d
                plot_data = single_linear_displacement_2d(
                    self.u, params['nelxyz'][0], params['nelxyz'][1], params['disp_factor']
                )
            self.plot_single_displacement(plot_data, is_3d_mode)
            
            self.status_bar.showMessage("Single displacement plot shown.", 3000)
        else:
            self.footer.create_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.status_bar.showMessage("Starting displacement computation...")
            
            self.progress_bar.setRange(0, params['disp_iterations']+1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            self.displacement_worker = DisplacementWorker(params, self.xPhys)
            self.displacement_worker.progress.connect(self.update_displacement_progress)
            self.displacement_worker.frameReady.connect(self.update_animation_frame)
            self.displacement_worker.finished.connect(self.handle_displacement_finished)
            self.displacement_worker.error.connect(self.handle_displacement_error)
            self.displacement_worker.start()
        
    def update_displacement_progress(self, iteration):
        """ Updates the progress bar and status message during displacement computation."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(f"Running non-linear displacement: step {iteration}...")

    def update_animation_frame(self, frame_data):
        """Updates the plot with a new frame from the displacement animation."""
        # Safety checks to ensure a plot exists and parameters are available
        if not self.figure.get_axes() or not self.last_params:
            return
        ax = self.figure.get_axes()[0]
        is_3d_mode = self.last_params['nelxyz'][2] > 0

        if is_3d_mode:
            # --- 3D Case: Clear and redraw the entire scene for each frame ---
            ax.clear()
            
            p = self.last_params
            nx, ny, nz = p['nelxyz']
            
            # Use the fast scatter plot with variable alpha for the "cloud" effect
            visible_elements_mask = frame_data > 0.01
            visible_indices = np.where(visible_elements_mask)[0]
            densities = frame_data[visible_indices]

            # Calculate coordinates for only the visible elements
            z = visible_indices // (nx * ny)
            x = (visible_indices % (nx * ny)) // ny
            y = visible_indices % ny
            
            # Create the RGBA color array where alpha = density
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            colors = np.zeros((len(densities), 4))
            colors[:, :3] = base_color_rgb
            colors[:, 3] = densities

            ax.scatter(x + 0.5, y + 0.5, z + 0.5, 
                       s=6000/max(nx, ny, nz),
                       marker='s', 
                       c=colors)
            
            # After clearing, we must redraw the overlays (forces, supports, frame, etc.)
            self._redraw_non_material_layers(ax, is_3d_mode=True)

        else:
            # --- 2D Case: Efficiently update the existing image data ---
            if not ax.images: return
            im = ax.images[0]
            
            nx, ny = self.last_params['nelxyz'][:2]
            im.set_array(frame_data.reshape((nx, ny)).T)
        
        # Redraw the canvas to show the changes
        self.canvas.draw()
        
    def on_displacement_preview_changed(self):
        """Triggers a replot if the preview is active when displacement factor changes."""
        if self.sections['displacement'].visibility_button.isChecked():
            self.replot()
            
    def handle_displacement_finished(self, message):
        """Handles the results after displacement computation finishes successfully."""
        self.status_bar.showMessage(message, 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        # After animation, replot the original result
        self.replot()

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

    def save_as_png(self):
        """Saves the current canvas to a file."""
        if self.xPhys is None: return
        
        if not os.path.exists('results'):
            os.makedirs('results')
            
        filename = 'results/result_2d.png'
        if self.last_params['nelxyz'][2] > 0:
            filename = 'results/result_3d.png'
        
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_bar.showMessage(f"Result saved to {filename}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the file:\n{e}")

    def save_as_vti(self):
        """Saves the current 3D result as a .vti file."""
        if self.xPhys is None:
            QMessageBox.warning(self, "Export Error", "You must run an optimization before exporting.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save as VTI", "results/result.vti", "VTK Image Data (*.vti)")
        if not filename: return
         
        success, error_msg = exporters.save_as_vti(self.xPhys, self.last_params['nelxyz'], filename)
        if success:
            self.status_bar.showMessage(f"Result saved to {filename}", 5000)
        else:
            QMessageBox.critical(self, "Save Error", f"Could not save VTI file:\n{error_msg}")

    def save_as_stl(self):
        """Saves the current 3D result as an .stl file."""
        if self.xPhys is None:
            QMessageBox.warning(self, "Export Error", "You must run an optimization before exporting.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save as STL", "results/result.stl", "STL Files (*.stl)")
        if not filename: return
        
        success, error_msg = exporters.save_as_stl(self.xPhys, self.last_params['nelxyz'], filename)
        if success:
            self.status_bar.showMessage(f"Result saved to {filename}", 5000)
        else:
            QMessageBox.critical(self, "Save Error", f"Could not save STL file:\n{error_msg}")

    ########
    # PLOT #
    ########

    def set_initial_plot(self):
        """Sets the initial plot to a default state with no data."""
        self.figure.clear()
        
        # Get theme colors
        bg_color = '#2E2E2E' if self.current_theme == 'dark' else '#F0F0F0'
        text_color = 'white' if self.current_theme == 'dark' else 'black'
        self.figure.patch.set_facecolor(bg_color)

        ax = self.figure.add_subplot(111, facecolor=bg_color)
        
        ax.text(0.5, 0.5, 'Configure parameters and press "Create"',
                ha='center', va='center', fontsize=16, alpha=0.5, color=text_color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
        
    def style_plot_default(self):
        """Sets the plot to a fixed white theme. Called only once."""
        self.figure.patch.set_facecolor('white')
        if self.figure.get_axes():
            ax = self.figure.get_axes()[0]
            ax.set_facecolor('white')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
        self.canvas.draw()
        
    def replot(self):
        """Redraws the plot canvas, intelligently showing or hiding each layer based on the state of the visibility buttons."""
        if not self.last_params:
            return # Do nothing if triggerd in sections initialization
        
        self.figure.clear()
        self.figure.patch.set_facecolor('white')
        
        is_3d_mode = self.last_params['nelxyz'][2] > 0 if self.last_params else False
        
        if is_3d_mode:
            ax = self.figure.add_subplot(111, projection='3d', facecolor='white')
        else:
            ax = self.figure.add_subplot(111, facecolor='white')

        # Layer 1: The Main Result (Material)
        if self.xPhys is not None and self.sections['material'].visibility_button.isChecked():
            self.plot_material(ax, is_3d=is_3d_mode)
        elif self.xPhys is None:
            if is_3d_mode:
                p = self.last_params
                self.xPhys = np.full(p['nelxyz'][0] * p['nelxyz'][1] * p['nelxyz'][2], p['volfrac'])
                if p['v'] != '-':
                    nelx, nely, nelz = p['nelxyz'][0], p['nelxyz'][1], p['nelxyz'][2]
                    cx, cy, cz = p['c'][0], p['c'][1], p['c'][2]
                    if p['v'] == '□':  # Cube
                        x_min, x_max = max(0, int(cx - p['r'])), min(nelx, int(cx + p['r']))
                        y_min, y_max = max(0, int(cy - p['r'])), min(nely, int(cy + p['r']))
                        z_min, z_max = max(0, int(cz - p['r'])), min(nelz, int(cz + p['r']))

                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        idx_z = np.arange(z_min, z_max)

                        if len(idx_x) > 0 and len(idx_y) > 0 and len(idx_z) > 0:
                            xx, yy, zz = np.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
                            indices = zz + yy * nelz + xx * nely * nelz
                            self.xPhys[indices.flatten()] = 1e-6

                    elif p['v'] == '○':  # Sphere
                        x_min, x_max = max(0, int(cx - p['r'])), min(nelx, int(cx + p['r']) + 1)
                        y_min, y_max = max(0, int(cy - p['r'])), min(nely, int(cy + p['r']) + 1)
                        z_min, z_max = max(0, int(cz - p['r'])), min(nelz, int(cz + p['r']) + 1)

                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        idx_z = np.arange(z_min, z_max)

                        if len(idx_x) > 0 and len(idx_y) > 0 and len(idx_z) > 0:
                            i_grid, j_grid, k_grid = np.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
                            mask = (i_grid - cx)**2 + (j_grid - cy)**2 + (k_grid - cz)**2 <= p['r']**2
                            ii, jj, kk = i_grid[mask], j_grid[mask], k_grid[mask]
                            indices = kk + jj * nelz + ii * nely * nelz
                            self.xPhys[indices] = 1e-6
                ax.text(0.5, 0.5, 0.5, s='Configure parameters and press "Create"', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, alpha=0.5, color='black')
            else:
                p = self.last_params
                self.xPhys = np.full(p['nelxyz'][0] * p['nelxyz'][1], p['volfrac'])
                if p['v'] != '-':
                    nelx, nely = p['nelxyz'][0], p['nelxyz'][1]
                    cx, cy = p['c'][0], p['c'][1]
            
                    if p['v'] == '□':  # Square
                        x_min, x_max = max(0, int(cx - p['r'])), min(nelx, int(cx + p['r']))
                        y_min, y_max = max(0, int(cy - p['r'])), min(nely, int(cy + p['r']))
                        
                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        if len(idx_x) > 0 and len(idx_y) > 0:
                            xx, yy = np.meshgrid(idx_x, idx_y, indexing='ij')
                            indices = (yy + xx * nely).flatten()
                            self.xPhys[indices] = 1e-6

                    elif p['v'] == '○':  # Circle
                        x_min, x_max = max(0, int(cx - p['r'])), min(nelx, int(cx + p['r']) + 1)
                        y_min, y_max = max(0, int(cy - p['r'])), min(nely, int(cy + p['r']) + 1)
                        
                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        if len(idx_x) > 0 and len(idx_y) > 0:
                            i_grid, j_grid = np.meshgrid(idx_x, idx_y, indexing='ij')
                            mask = (i_grid - cx)**2 + (j_grid - cy)**2 <= p['r']**2
                            ii, jj = i_grid[mask], j_grid[mask]
                            indices = jj + ii * nely
                            self.xPhys[indices] = 1e-6
                ax.text(0.5, 0.5, s='Configure parameters and press "Create"', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, alpha=0.5, color='black')
            self.plot_material(ax, is_3d=is_3d_mode)
        
        self.redraw_non_material_layers(ax, is_3d_mode)
        self.canvas.draw()
    
    #def plot_3d_layers_fast(self, ax): # call below for fast 3D plotting (bad quality when result isn't sharp)
    #    """
    #    Helper function to plot the 3D material layer using a fast isosurface.
    #    """
    #    p = self.last_params
    #    if self.xPhys is None:
    #        return
    #    
    #    nx, ny, nz = p['nelxyz']
    #    
    #    # 1. Reshape the density field. Note the order (nz, nx, ny) for mcubes.
    #    x_phys_3d = self.xPhys.reshape((nz, nx, ny))
    #    
    #    # 2. Use the fast Marching Cubes algorithm to extract the isosurface
    #    #    at a 0.5 density level. This is the core of the speed improvement.
    #    try:
    #        vertices, triangles = mcubes.marching_cubes(x_phys_3d, 0.5) 
    #    except ValueError:
    #        # This can happen if the density field is completely empty.
    #        # In this case, we have nothing to plot.
    #        print("Warning: Marching cubes found no surface to plot.")
    #        return
    #    
    #    # Check that something was extracted
    #    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
    #        print("Warning: Empty mesh from marching cubes.")
    #        return
    #    
    #    # 3. Get the material color for plotting
    #    color = self.material_widget.mat_color.get_color()
    #    
    #    # 4. Plot the extracted surface as a single, efficient triangular mesh.
    #    #    The vertices are returned in (Z, X, Y) order, so we pass them to
    #    #    plot_trisurf in the correct (X, Y, Z) order.
    #    ax.plot_trisurf(
    #        vertices[:, 1],  # X coordinates
    #        vertices[:, 2],  # Y coordinates
    #        triangles,       # The connections between vertices
    #        vertices[:, 0],  # Z coordinates
    #        color=color,
    #        edgecolor=(0, 0, 0, 0.1), # Faint black edges
    #        linewidth=0.1
    #    )
    #    
    #    # Set the aspect ratio to match the domain dimensions
    #    ax.set_box_aspect([nx, ny, nz]) # Correct scaling
    
    def plot_material(self, ax, is_3d, xPhys_data = None):
        """Plot the material."""
        p = self.last_params
        nx, ny, nz = p['nelxyz'][0], p['nelxyz'][1], p['nelxyz'][2]
        data_to_plot = xPhys_data if xPhys_data is not None else self.xPhys
        if data_to_plot is None:
            return
        
        if is_3d:
            ax.clear()
            # Plot using voxels -> only the exterior box is visible
            #x_phys_3d = data_to_plot.reshape((nz, nx, ny)).transpose(1, 2, 0) if xPhys_data is None else xPhys_data.reshape((nz, nx, ny)).transpose(1, 2, 0)
            #base_color = np.array(to_rgb(self.material_widget.mat_color.get_color()))
            #color = np.zeros(x_phys_3d.shape + (4,))
            #color[..., :3] = base_color  # Set RGB
            #color[..., 3] = np.clip(x_phys_3d, 0.0, 1.0)
            #ax.voxels(x_phys_3d, facecolors=color, edgecolor=None, ) # Very slow for large grids
            #ax.set_box_aspect([nx, ny, nz])
            #self.redraw_non_material_layers(ax, is_3d_mode=True)
            
            p = self.last_params
            nx, ny, nz = p['nelxyz']
            
            # Find all elements with a small amount of material (e.g., > 1%)
            # This avoids plotting millions of fully transparent points.
            visible_elements_mask = data_to_plot > 0.01
            visible_indices = np.where(visible_elements_mask)[0]
            
            densities = data_to_plot[visible_indices]

            z = visible_indices // (nx * ny)
            x = (visible_indices % (nx * ny)) // ny
            y = visible_indices % ny
            
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            
            colors = np.zeros((len(densities), 4))
            colors[:, :3] = base_color_rgb  # Set the RGB color for all points
            colors[:, 3] = densities      # Set the Alpha channel to the density

            ax.scatter(x + 0.5, y + 0.5, z + 0.5, 
                       s=6000/max(nx, ny, nz),
                       marker='s', 
                       c=colors, # Pass the array of colors with variable alpha
                       alpha=None) # Alpha is now controlled by the 'c' array
            
            ax.set_box_aspect([nx, ny, nz])
            self.redraw_non_material_layers(ax, is_3d_mode=True)
        else:
            mat_color = self.material_widget.mat_color.get_color()
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", mat_color])
            
            ax.imshow(data_to_plot.reshape((nx, ny)).T, cmap=cmap, interpolation='nearest',
                    origin='lower', norm=plt.Normalize(0, 1))

    def redraw_non_material_layers(self, ax, is_3d_mode):
        """Helper to draw all the plot layers that are NOT the main result."""
        # Layer 2: Overlays
        self.plot_forces(ax, is_3d=is_3d_mode)
        self.plot_supports(ax, is_3d=is_3d_mode)
        self.plot_void_region(ax, is_3d=is_3d_mode)
        self.plot_dimensions_frame(ax, is_3d=is_3d_mode)
        self.plot_displacement_preview(ax, is_3d=is_3d_mode)

        # Layer 3: Dimension Labels and Ticks
        if self.sections['dimensions'].visibility_button.isChecked():
            ax.set_xlabel("X", color='black')
            ax.set_ylabel("Y", color='black')
            if is_3d_mode: ax.set_zlabel("Z", color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            if is_3d_mode: ax.tick_params(axis='z', colors='black')
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor('black')
        else:
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xticks([]); ax.set_yticks([])
            if is_3d_mode: ax.set_zticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Final adjustments
        if not is_3d_mode:
            ax.set_aspect('equal', 'box')
        ax.autoscale(tight=True)

    def plot_dimensions_frame(self, ax, is_3d):
        """Draws a dotted frame around the design space, controlled by the Dimensions section's visibility button."""
        if not self.sections['dimensions'].visibility_button.isChecked():
            return

        p = self.last_params
        nx, ny, nz = p['nelxyz']

        if is_3d:
            # Define the 8 vertices of the box
            verts = [
                (0, 0, 0), (nx, 0, 0), (nx, ny, 0), (0, ny, 0),
                (0, 0, nz), (nx, 0, nz), (nx, ny, nz), (0, ny, nz)
            ]
            # Define the 12 edges by connecting the vertices
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for edge in edges:
                points = [verts[edge[0]], verts[edge[1]]]
                x, y, z = zip(*points)
                ax.plot(x, y, z, color='gray', linestyle=':', linewidth=1.5)
        else:
            rect = Rectangle((0, 0), nx, ny, fill=False, edgecolor='gray', linestyle=':', linewidth=1.5)
            ax.add_patch(rect)

    def plot_forces(self, ax, is_3d):
        """Plots the forces as arrows."""
        if not self.sections['forces'].visibility_button.isChecked(): return
        p = self.last_params
        colors = ['r', 'b', 'b']
        for i in range(3):
            if p['a'][i] != '-':
                direction = p['a'][i]
                length = np.mean(p['nelxyz'][:2])/6
                
                dx, dy, dz = 0, 0, 0
                if 'X:→' in direction: dx=length
                if 'X:←' in direction: dx=-length
                if 'Y:↑' in direction: dy=length
                if 'Y:↓' in direction: dy=-length
                if 'Z:<' in direction: dz=length
                if 'Z:>' in direction: dz=-length
                
                if is_3d:
                    ax.quiver(p['fx'][i], p['fy'][i], p['fz'][i], dx, dy, dz, color=colors[i], length=length, normalize=True)
                else:
                    ax.quiver(p['fx'][i], p['fy'][i], dx, dy, color=colors[i], scale_units='xy', angles='xy', scale=1)

    def plot_supports(self, ax, is_3d):
        """Plots the supports as triangles."""
        if not self.sections['supports'].visibility_button.isChecked(): return
        p = self.last_params
        for i in range(len(p['dim'])):
            if p['dim'][i] != '-':
                pos = [p['sx'][i], p['sy'][i], p['sz'][i]]
                if is_3d:
                    ax.scatter(pos[0], pos[1], pos[2], s=80, marker='^', c='black', depthshade=False)
                else:
                    ax.scatter(pos[0], pos[1], s=80, marker='^', c='black')

    def plot_void_region(self, ax, is_3d):
        """Plots the void region outline (square/cube or circle/sphere) in 2D or 3D."""
        if not self.sections['void'].visibility_button.isChecked():
            return
        
        p = self.last_params
        if not p or p['v'] == '-':
            return # Do nothing if params don't exist or no shape is selected

        r, c = p['r'], p['c']

        if is_3d:
            if p['v'] == '□':
                cx, cy, cz = c
                # Define the 8 vertices of the cube
                verts = np.array([
                    [cx-r, cy-r, cz-r], [cx+r, cy-r, cz-r], [cx+r, cy+r, cz-r], [cx-r, cy+r, cz-r],
                    [cx-r, cy-r, cz+r], [cx+r, cy-r, cz+r], [cx+r, cy+r, cz+r], [cx-r, cy+r, cz+r]
                ])
                # Define the 12 edges connecting the vertices
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                for edge in edges:
                    points = verts[list(edge)]
                    # Note: Matplotlib's 3D axes are ordered (X, Y, Z)
                    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', linestyle=':')

            elif p['v'] == '○':
                cx, cy, cz = c
                # Create the surface grid for the sphere
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                # Parametric equations for a sphere
                x = cx + r * np.outer(np.cos(u), np.sin(v))
                y = cy + r * np.outer(np.sin(u), np.sin(v))
                z = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_wireframe(x, y, z, color='green', linestyle=':')
        
        else:
            if p['v'] == '□':
                rect = plt.Rectangle((c[0]-r, c[1]-r), 2*r, 2*r, fill=False, edgecolor='green', linestyle=':')
                ax.add_patch(rect)
            elif p['v'] == '○':
                circ = plt.Circle((c[0], c[1]), r, fill=False, edgecolor='green', linestyle=':')
                ax.add_patch(circ)

    def show_blank_plot(self):
        """Clears the canvas and displays a blank white plot."""
        self.figure.clear()
        self.figure.patch.set_facecolor('white')
        ax = self.figure.add_subplot(111, facecolor='white')
        
        # Ensure there are no ticks or labels
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False) # Hide the black border frame
        
        self.canvas.draw()
        
    def plot_single_displacement(self, plot_data, is_3d):
        """Draws the deformed voxel grid from a single-frame analysis."""
        if is_3d:
            nelx, nely, nelz = self.last_params['nelxyz']

            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d', facecolor='white')
            self.figure.patch.set_facecolor('white')

            x_phys_3d = self.xPhys.reshape((nelz, nelx, nely))

            # Use alpha = density to visualize deformation
            base_color = np.array(to_rgb(self.material_widget.mat_color.get_color()))
            colors = np.zeros(x_phys_3d.shape + (4,))
            colors[..., :3] = base_color
            colors[..., 3] = np.clip(x_phys_3d, 0.0, 1.0)

            # Optional: downsample or use only surface for performance
            mask = x_phys_3d > 1e-3  # Optional surface mask logic here

            ax.voxels(mask, facecolors=colors, edgecolor=None)
            ax.set_box_aspect([nelz, nelx, nely])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            X, Y = plot_data
            nelx, nely = self.last_params['nelxyz'][:2]
            self.figure.clear(); ax = self.figure.add_subplot(111, facecolor='white')
            self.figure.patch.set_facecolor('white')
            ax.pcolormesh(X, Y, -self.xPhys.reshape((nelx, nely)), cmap='gray') # , shading='auto'
            ax.set_aspect('equal', 'box'); ax.autoscale(tight=True)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        
        self.canvas.draw()

    def plot_displacement_preview(self, ax, is_3d):
        """Overlays displacement vectors (quivers) on the plot if the preview is active."""
        if not self.sections['displacement'].visibility_button.isChecked():
            return
        if self.u is None or self.xPhys is None:
            return

        p = self.last_params
        nelx, nely = p['nelxyz'][:2]
        disp_factor = self.displacement_widget.mov_disp.value()

        if is_3d:
            from app.displacements.displacement_3d import get_edofMat, get_element_displacements
            nelx, nely, nelz = p['nelxyz']
            step = 5 # Draw a vector every 5 elements to avoid clutter

            # 1. Get average displacement per element
            edofMat = get_edofMat(nelx, nely, nelz)
            ux, uy, uz = get_element_displacements(self.u, edofMat)

            # 2. Create a grid of element coordinates to sample from
            elx, ely, elz = np.meshgrid(np.arange(0, nelx, step), 
                                        np.arange(0, nely, step),
                                        np.arange(0, nelz, step), indexing='ij')

            # 3. Filter for points that are inside the material
            el_indices = elz * (nelx * nely) + elx * nely + ely
            material_mask = self.xPhys[el_indices] > 0.5
            
            # 4. Get the coordinates and displacement vectors for the valid points
            x_valid = elx[material_mask] + 0.5 # Center of element
            y_valid = ely[material_mask] + 0.5
            z_valid = elz[material_mask] + 0.5
            el_indices_valid = el_indices[material_mask]

            ax.quiver(x_valid, y_valid, z_valid,
                      ux[el_indices_valid] * disp_factor,
                      uy[el_indices_valid] * disp_factor,
                      uz[el_indices_valid] * disp_factor,
                      color='red', length=disp_factor/4, normalize=True)
        else:
            # Create a grid of points to draw vectors at (e.g., every 5th element)
            step = 5
            i_coords, j_coords = np.meshgrid(np.arange(0, nelx, step), np.arange(0, nely, step), indexing='ij')
            
            # Get the 1D indices for the original xPhys and u arrays
            el_indices = (i_coords * nely + j_coords).flatten()
            node_indices = (i_coords * (nely + 1) + j_coords).flatten()
            
            # Filter for points that are inside the material
            material_mask = self.xPhys[el_indices] > 0.5
            
            # Get the coordinates and displacement vectors for the valid points
            i_valid = i_coords.flatten()[material_mask]
            j_valid = j_coords.flatten()[material_mask]
            node_valid = node_indices[material_mask]
            
            factor = disp_factor / p['fv'][0] if p['fv'][0] != 0 else disp_factor
            ux = self.u[2 * node_valid, 0] * factor
            uy = -self.u[2 * node_valid + 1, 0] * factor
            
            ax.quiver(i_valid, j_valid, ux, uy, color='red', scale=40, scale_units='xy', angles='xy')

    ###########
    # Presets #
    ###########

    def load_presets(self):
        """Loads presets from the JSON file and populates the combo box."""
        try:
            with open(self.presets_file, 'r') as f:
                self.presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.presets = {}
            print("Presets file not found or invalid. Starting fresh.")

        self.preset.presets_combo.blockSignals(True)
        self.preset.presets_combo.clear()
        self.preset.presets_combo.addItem("Select a preset...") # Index 0
        self.preset.presets_combo.addItems(sorted(self.presets.keys()))
        self.preset.presets_combo.blockSignals(False)
        self.preset.delete_preset_button.setEnabled(False)

    def save_presets(self):
        """Saves the current presets dictionary to the JSON file."""
        with open(self.presets_file, 'w') as f:
            json.dump(self.presets, f, indent=4)
        self.status_bar.showMessage("Presets saved.", 3000)

    def save_new_preset(self):
        """Prompts the user for a name and saves the current parameters."""
        preset_name, ok = QInputDialog.getText(self, "Save Preset", "Enter a name for this preset:")
        
        if ok and preset_name:
            if preset_name in self.presets:
                reply = QMessageBox.question(self, "Overwrite Preset", 
                                             f"A preset named '{preset_name}' already exists. Overwrite it?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return # User cancelled

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

        reply = QMessageBox.warning(self, "Delete Preset",
                                    f"You are about to permanently delete the preset '{preset_name}'.\nContinue?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            del self.presets[preset_name]
            self.save_presets()
            self.load_presets() # Reloads the list and disables the delete button
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
        # --- Block signals to prevent a cascade of replots ---
        all_widgets = [self.dim_widget.nx, self.dim_widget.ny, self.dim_widget.nz,
                       self.dim_widget.volfrac, self.void_widget.v_shape, self.void_widget.v_radius,
                       self.void_widget.v_cx, self.void_widget.v_cy, self.void_widget.v_cz,
                       self.material_widget.mat_E, self.material_widget.mat_nu,
                       self.optimizer_widget.opt_ft, self.optimizer_widget.opt_fr,
                       self.optimizer_widget.opt_p, self.optimizer_widget.opt_n_it]
        for w in all_widgets: w.blockSignals(True)
        for group in self.forces_widget.inputs + self.supports_widget.inputs:
            for w in group.values(): w.blockSignals(True)

        # Apply values
        
        # Dimensions
        self.dim_widget.nx.setValue(params['nelxyz'][0])
        self.dim_widget.ny.setValue(params['nelxyz'][1])
        self.dim_widget.nz.setValue(params['nelxyz'][2])
        self.dim_widget.volfrac.setValue(params.get('volfrac', 0.3))
        self.update_position_ranges()
        self.on_mode_changed()
        
        # Void Region
        self.void_widget.v_shape.setCurrentText(f"{params['v']} (Square)" if params['v'] == '□' else f"{params['v']} (Circle)" if params['v'] == '○' else '-')
        self.void_widget.v_radius.setValue(params['r'])
        self.void_widget.v_cx.setValue(params['c'][0])
        self.void_widget.v_cy.setValue(params['c'][1])
        self.void_widget.v_cz.setValue(params['c'][2])
        
        # Forces
        for i, group in enumerate(self.forces_widget.inputs):
            group['fx'].setValue(params['fx'][i]); group['fy'].setValue(params['fy'][i]); group['fz'].setValue(params['fz'][i])
            group['a'].setCurrentText(params['a'][i]); group['fv'].setValue(params['fv'][i])
        
        # Supports
        num_supports_in_preset = len(params.get('sx', []))
        for i, support_group in enumerate(self.supports_widget.inputs):
            if i < num_supports_in_preset:
                # If data exists for this support, apply it
                support_group['sx'].setValue(params['sx'][i])
                support_group['sy'].setValue(params['sy'][i])
                support_group['sz'].setValue(params['sz'][i])
                support_group['d'].setCurrentText(params['dim'][i])
            else:
                # If no data exists, reset this row to default empty values
                support_group['sx'].setValue(0)
                support_group['sy'].setValue(0)
                support_group['sz'].setValue(0)
                support_group['d'].setCurrentIndex(0) # Set to '-'

        # Material
        self.material_widget.mat_E.setValue(params['E'])
        self.material_widget.mat_nu.setValue(params['nu'])

        # Optimizer
        self.optimizer_widget.opt_ft.setCurrentIndex(0 if params['ft'] == "Sensitivity" else 1)
        self.optimizer_widget.opt_fr.setValue(params['rmin'])
        self.optimizer_widget.opt_p.setValue(params['penal'])
        self.optimizer_widget.opt_n_it.setValue(params['n_it'])
        
        # Displacement
        self.displacement_widget.mov_disp.setValue(params['disp_factor'])
        self.displacement_widget.mov_iter.setValue(params['disp_iterations'])
        
        # unblock signals
        for w in all_widgets: w.blockSignals(False)
        for group in self.forces_widget.inputs + self.supports_widget.inputs:
            for w in group.values(): w.blockSignals(False)
        
        # Manually trigger a single update
        self.update_position_ranges()
        self.on_parameter_changed()
        self.status_bar.showMessage(f"Loaded preset: {self.preset.presets_combo.currentText()}", 3000)

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
        stylesheet = LIGHT_THEME_STYLESHEET if theme_name == 'light' else DARK_THEME_STYLESHEET
        QApplication.instance().setStyleSheet(stylesheet)
        self.current_theme = theme_name
        icons.set_theme(theme_name) 
        
        # On the very first run, we need to style the plot once.
        # Afterwards, the plot theme is never touched again.
        if initial_setup:
            self.style_plot_default()

    def toggle_theme(self):
        """Switches between light and dark themes."""
        if self.current_theme == 'light':
            self.set_theme('dark')
        else:
            self.set_theme('light')
        self.update_ui_icons()

    def update_ui_icons(self):
        """
        Resets all icons in the UI to force them to be re-fetched from the
        now theme-aware IconProvider.
        """
        # Update the theme button itself
        if self.current_theme == 'dark':
            self.header.theme_button.setIcon(icons.get('sun'))
            self.header.theme_button.setToolTip("Switch to Light Theme")
        else:
            self.header.theme_button.setIcon(icons.get('moon'))
            self.header.theme_button.setToolTip("Switch to Dark Theme")

        # Update other static icons
        self.preset.save_preset_button.setIcon(icons.get('save'))
        self.preset.delete_preset_button.setIcon(icons.get('delete'))
        self.header.info_button.setIcon(icons.get('info'))
        self.footer.create_button.setIcon(icons.get('create'))
        self.footer.save_button.setIcon(icons.get('save'))
        self.displacement_widget.run_disp_button.setIcon(icons.get('move'))

        # Update dynamic icons (like the visibility and collapsible arrows)
        for section in self.sections.values():
            # This requires a new method on the CollapsibleSection widget
            section.update_all_icons()

    def open_github_link(self):
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/Topopt-Comec") 
        QDesktopServices.openUrl(url)