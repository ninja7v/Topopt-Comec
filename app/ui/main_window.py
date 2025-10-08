# app/ui/main_window.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Main window for the Topopt Comec application using PySide6.

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
from typing import List
#import mcubes

from app.ui import exporters
from app.core import initializers
from .workers import OptimizerWorker, DisplacementWorker
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
        self.sections['void'] = self.create_voids_section()
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
        self.dim_widget.scale_button.clicked.connect(self.scale_parameters)
        self.dim_widget.nx.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.ny.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self.update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self.on_mode_changed)
        return section

    def create_voids_section(self):
        """Creates the second section for void region parameters."""
        self.void_widget = VoidWidget()
        section = CollapsibleSection("⚫ Void Region", self.void_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        for void_group in self.void_widget.inputs:
            void_group['vshape'].currentIndexChanged.connect(self.on_parameter_changed)
            void_group['vradius'].valueChanged.connect(self.on_parameter_changed)
            void_group['vx'].valueChanged.connect(self.on_parameter_changed)
            void_group['vy'].valueChanged.connect(self.on_parameter_changed)
            void_group['vz'].valueChanged.connect(self.on_parameter_changed)
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
            force_group['fdir'].currentIndexChanged.connect(self.on_parameter_changed)
            force_group['fnorm'].valueChanged.connect(self.on_parameter_changed)
            
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
            support_input_group['sdim'].currentIndexChanged.connect(self.on_parameter_changed)
            
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
        self.material_widget.mat_init_type.currentIndexChanged.connect(self.on_parameter_changed)
        return section

    def create_optimizer_section(self):
        """Creates the sixth section for optimization parameters."""
        self.optimizer_widget = OptimizerWidget()
        section = CollapsibleSection("💻 Optimizer", self.optimizer_widget)
        self.optimizer_widget.opt_ft.currentIndexChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_fr.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_p.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_max_change.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_n_it.valueChanged.connect(self.on_parameter_changed)
        return section
    
    def create_displacement_section(self):
        """Creates the seventh section for displacement animation parameters."""
        self.displacement_widget = DisplacementWidget()
        section = CollapsibleSection("↔️ Displacement", self.displacement_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self.on_visibility_toggled)
        section.visibility_button.setEnabled(False) # Disabled until a result is ready
        self.displacement_widget.run_disp_button.setEnabled(False) # Disabled until a result is ready
        section.visibility_button.setToolTip("Preview displacement vectors on the main plot")
        section.visibility_button.setChecked(False)
        
        section.visibility_button.toggled.connect(self.replot)
        self.displacement_widget.run_disp_button.clicked.connect(self.run_displacement)
        self.displacement_widget.stop_disp_button.clicked.connect(self.stop_displacement)
        self.displacement_widget.reset_disp_button.clicked.connect(self.reset_displacement_view)
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

        # Void regions
        params['vshape'], params['vradius'] = [], []
        params['vx'], params['vy'], params['vz'] = [], [], []
        for vw in self.void_widget.inputs:
            params['vshape'].append(vw['vshape'].currentText()[0])
            params['vradius'].append(vw['vradius'].value())
            params['vx'].append(vw['vx'].value())
            params['vy'].append(vw['vy'].value())
            params['vz'].append(vw['vz'].value())
        
        # Forces
        params['fx'], params['fy'], params['fz'] = [], [], []
        params['fdir'], params['fnorm'] = [], []
        for fw in self.forces_widget.inputs:
            params['fx'].append(fw['fx'].value())
            params['fy'].append(fw['fy'].value())
            params['fz'].append(fw['fz'].value())
            params['fdir'].append(fw['fdir'].currentText())
            params['fnorm'].append(fw['fnorm'].value())
            
        # Supports
        params['sx'], params['sy'], params['sz'] = [], [], []
        params['sdim'] = []
        for sw in self.supports_widget.inputs:
            params['sx'].append(sw['sx'].value())
            params['sy'].append(sw['sy'].value())
            params['sz'].append(sw['sz'].value())
            params['sdim'].append(sw['sdim'].currentText())
            
        # Material
        params['E'] = self.material_widget.mat_E.value()
        params['nu'] = self.material_widget.mat_nu.value()
        params['init_type'] = self.material_widget.mat_init_type.currentIndex()
        
        # Optimizer
        params['filter_type'] = 'Sensitivity' if self.optimizer_widget.opt_ft.currentIndex() == 0 else 'Density'
        params['filter_radius_min'] = self.optimizer_widget.opt_fr.value()
        params['penal'] = self.optimizer_widget.opt_p.value()
        params['max_change'] = self.optimizer_widget.opt_max_change.value()
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
            self.is_displaying_deformation = False

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
            # --- Normalize Void ---
            if 'vshape' in p:
                zipped_voids = zip(p.get('vshape', []), p.get('vradius', []), p.get('vx', []), p.get('vy', []), p.get('vz', []) if not is_2d else [0]*len(p.get('vx', [])))
                void_list = list(zipped_voids)
                active_voids = [v for v in void_list if v[0] != '-']
                if active_voids:
                    vshape, vradius, vx, vy, vz = list(zip(*active_voids))
                    p['vshape'], p['vradius'], p['vx'], p['vy'], p['vz'] = list(vshape), list(vradius), list(vx), list(vy), list(vz)
                else:
                    for key in ['vshape', 'vradius', 'vx', 'vy', 'vz']:
                        p.pop(key, None) # pop them, not just empty them
                if is_2d and 'vz' in p:
                    p.pop('vz')
            # --- Normalize Supports ---
            if 'sdim' in p:
                zipped_supports = zip(p.get('sx', []), p.get('sy', []), p.get('sz', []) if not is_2d else [0]*len(p.get('sx', [])), p.get('sdim', []))
                active_supports = [s for s in zipped_supports if s[3] != '-']
                if active_supports:
                    sx, sy, sz, sdim = list(zip(*active_supports))
                    p['sx'], p['sy'], p['sz'], p['sdim'] = list(sx), list(sy), list(sz), list(sdim)
                else: # Should not happen as at least one support is required
                    p['sx'], p['sy'], p['sz'], p['sdim'] = [], [], [], []
                if is_2d and 'sz' in p:
                    p.pop('sz')
            # --- Normalize Forces ---
            if 'fdir' in p:
                zipped_forces = zip(p.get('fx', []), p.get('fy', []), p.get('fz', []) if not is_2d else [0]*len(p.get('fx', [])), p.get('fdir', []), p.get('fnorm', []))
                force_list = list(zipped_forces)
                # Keep the input force, and any output forces that are active.
                active_forces = [force_list[0]] + [f for f in force_list[1:] if f[3] != '-']
                if active_forces:
                    fx, fy, fz, fdir, fnorm = list(zip(*active_forces))
                    p['fx'], p['fy'], p['fz'], p['fdir'], p['fnorm'] = list(fx), list(fy), list(fz), list(fdir), list(fnorm)
                else: # Should not happen as one input force is required
                    p['fx'], p['fy'], p['fz'], p['fdir'], p['fnorm'] = [], [], [], [], []
                if is_2d and 'fz' in p:
                    p.pop('fz')
            
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
        if p['fdir'][0] == '-': return "Input force (Force 1) direction must be set."
        if len(p['fdir']) > 2 and p['fdir'][1] == '-' and p['fdir'][2] == '-': 
            return "At least one output force must be set."
        elif len(p['fdir']) == 2 and p['fdir'][1] == '-':
            return "At least one output force must be set."
        elif len(p['fdir']) == 1:
            return "At least one output force must be set."
        has_support = any(d != '-' for d in p['sdim'])
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

        # Update ranges for all voids
        for void_group in self.void_widget.inputs:
            void_group['vx'].setMaximum(nx)
            void_group['vy'].setMaximum(ny)
            void_group['vz'].setMaximum(nz)

        # Update ranges for all forces
        for force_group in self.forces_widget.inputs:
            force_group['fx'].setMaximum(nx)
            force_group['fy'].setMaximum(ny)
            force_group['fz'].setMaximum(nz)

        # Update ranges for all supports
        for support_group in self.supports_widget.inputs:
            support_group['sx'].setMaximum(nx)
            support_group['sy'].setMaximum(ny)
            support_group['sz'].setMaximum(nz)
    
    def scale_parameters(self):
        """Scales all dimensional and positional parameters by a given factor."""
        scale = self.dim_widget.scale.value()
        is_3d = self.dim_widget.nz.value() > 0

        if scale == 1.0:
            self.status_bar.showMessage("Scale is 1.0, nothing to do.", 3000)
            return
        
        def check(value, scale):
            scaled_val = value * scale
            if (scaled_val < 1 or scaled_val >1000) and value > 0:
                return True, False
            if abs(scaled_val - round(scaled_val)) > 1e-6: # Check if it's not an integer
                return False, True
            return False, False

        proceed_impossible, warn_needed = False, False
        
        # Check dimensions
        for dim_widget in [self.dim_widget.nx, self.dim_widget.ny] + ([self.dim_widget.nz] if is_3d else []):
            pi, wn = check(dim_widget.value(), scale)
            proceed_impossible |= pi; warn_needed |= wn
        
        # Check void regions
        for void_group in self.void_widget.inputs:
            if void_group['vshape'].currentText() == '-': continue
            for key in ['vx', 'vy'] + (['vz'] if is_3d else []):
                pi, wn = check(void_group[key].value(), scale)
                proceed_impossible |= pi; warn_needed |= wn
            pi, wn = check(void_group['vradius'].value(), scale)
            proceed_impossible |= pi; warn_needed |= wn

        # Check forces
        for force_group in self.forces_widget.inputs:
            if force_group['fdir'].currentText() == '-': continue
            for key in ['fx', 'fy'] + (['fz'] if is_3d else []):
                pi, wn = check(force_group[key].value(), scale)
                proceed_impossible |= pi; warn_needed |= wn

        # Check supports
        for support_group in self.supports_widget.inputs:
            if support_group['sdim'].currentText() == '-': continue
            for key in ['sx', 'sy'] + (['sz'] if is_3d else []):
                pi, wn = check(support_group[key].value(), scale)
                proceed_impossible |= pi; warn_needed |= wn

        if proceed_impossible:
            QMessageBox.critical(self, "Scaling Error", "Scaling would lead position(s) out of range.")
            return
        if warn_needed:
            reply = QMessageBox.question(self, "Scaling Warning", "Scaling would loss initial proportions due to rounding(s). Proceed?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        # --- Perform Scaling ---
        # Temporarily block signals to prevent multiple replots
        self.block_all_parameter_signals(True)

        self.dim_widget.nx.setValue(round(self.dim_widget.nx.value() * scale))
        self.dim_widget.ny.setValue(round(self.dim_widget.ny.value() * scale))
        if is_3d: self.dim_widget.nz.setValue(round(self.dim_widget.nz.value() * scale))
        
        if scale > 1.0:
            self.update_position_ranges() # Update max ranges before scaling positions otherwise they might get clamped
        
        for group in self.void_widget.inputs:
            int0 = group['vx'].value()
            int1 = round(group['vx'].value() * scale)
            group['vx'].setValue(round(group['vx'].value() * scale))
            group['vy'].setValue(round(group['vy'].value() * scale))
            if is_3d: group['vz'].setValue(round(group['vz'].value() * scale))
            group['vradius'].setValue(max(1, round(group['vradius'].value() * scale)))

        for group in self.forces_widget.inputs:
            group['fx'].setValue(round(group['fx'].value() * scale))
            group['fy'].setValue(round(group['fy'].value() * scale))
            if is_3d: group['fz'].setValue(round(group['fz'].value() * scale))

        for group in self.supports_widget.inputs:
            group['sx'].setValue(round(group['sx'].value() * scale))
            group['sy'].setValue(round(group['sy'].value() * scale))
            if is_3d: group['sz'].setValue(round(group['sz'].value() * scale))
            
        if scale < 1.0:
            self.update_position_ranges() # Update max ranges after scaling positions otherwise values might be clamped before scaling
        
        self.block_all_parameter_signals(False)
        
        # Manually trigger a single, final update
        self.on_parameter_changed()
        self.status_bar.showMessage(f"All parameters scaled by a factor of {scale}.", 3000)

    def block_all_parameter_signals(self, block: bool):
        """Helper to block or unblock signals for all parameter widgets."""
        # A helper to make the code cleaner. Add all your widgets to this list.
        all_widgets = [self.dim_widget.nx, self.dim_widget.ny, self.dim_widget.nz,
                       self.dim_widget.volfrac,
                       self.material_widget.mat_E, self.material_widget.mat_nu, self.material_widget.mat_init_type,
                       self.optimizer_widget.opt_ft, self.optimizer_widget.opt_fr,
                       self.optimizer_widget.opt_p,
                       self.optimizer_widget.opt_max_change, self.optimizer_widget.opt_n_it]
        for w in all_widgets: w.blockSignals(block)
        for group in self.void_widget.inputs + self.forces_widget.inputs + self.supports_widget.inputs:
            for w in group.values(): w.blockSignals(block)
    
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
        if self.xPhys is None or self.u is None:
            QMessageBox.warning(self, "Displacement Error", "You must run a successful optimization before analyzing movement.")
            return
        
        self.is_displaying_deformation = True
        self.last_displayed_frame_data = None
        self.displacement_widget.button_stack.setCurrentWidget(self.displacement_widget.stop_disp_button)
        self.footer.create_button.setEnabled(False)
        
        self.replot()
        QApplication.processEvents()

        params = self.gather_parameters()
        is_3d_mode = params['nelxyz'][2] > 0
        if params['disp_iterations'] == 1:
            # Run single-frame logic directly
            self.status_bar.showMessage("Calculating single displacement frame...")
            QApplication.processEvents() # Update UI
            
            if is_3d_mode:
                from app.core.displacements import single_linear_displacement_3d
                self.last_displayed_frame_data = single_linear_displacement_3d(self.xPhys, self.u, *params['nelxyz'], params['disp_factor'])
            else:
                from app.core.displacements import single_linear_displacement_2d
                self.last_displayed_frame_data = single_linear_displacement_2d(self.u, params['nelxyz'][0], params['nelxyz'][1], params['disp_factor'])
            self.replot()
            self.handle_displacement_finished("Single frame shown.")
            self.status_bar.showMessage("Single displacement plot shown.", 3000)
        else:
            self.footer.create_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.status_bar.showMessage("Starting displacement computation...")
            
            self.progress_bar.setRange(0, params['disp_iterations']+1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            self.displacement_worker = DisplacementWorker(params, self.xPhys, self.u)
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
        self.replot() # Redraw the original view
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.displacement_widget.button_stack.setCurrentWidget(self.displacement_widget.run_disp_button)

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
            colors = np.zeros((len(densities), 4))
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            colors[:, :3] = base_color_rgb
            colors[:, 3] = densities

            ax.scatter(x + 0.5, y + 0.5, z + 0.5,
                       s=6000/max(nx, ny, nz),
                       marker='s', # Square markers to mimic voxels
                       c=colors,
                       alpha=None) # Alpha is now controlled by the 'c' array
            
            self.redraw_non_material_layers(ax, is_3d_mode=True)

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
        self.displacement_widget.run_disp_button.setEnabled(False)
        self.displacement_widget.button_stack.setCurrentWidget(self.displacement_widget.reset_disp_button)
        self.displacement_widget.stop_disp_button.setText(" Stop") # Reset text for next run
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
            base_name = "result_3d" if self.last_params["nelxyz"][2] > 0 else "result_2d"

        # File dialog config
        filters = {
            "png": ("Save as PNG", "Portable Network Graphics (*.png)"),
            "vti": ("Save as VTI", "VTK Image Data (*.vti)"),
            "stl": ("Save as STL", "STL File (*.stl)")
        }

        window_title, extension_filter = filters[file_type]
        default_path = f"results/{base_name}.{file_type}"

        filepath, _ = QFileDialog.getSaveFileName(self, window_title, default_path, extension_filter)
        if not filepath:  # user canceled
            return

        try:
            if file_type == "png":
                self.figure.savefig(filepath, dpi=300, bbox_inches="tight")

            elif file_type == "vti":
                success, error_msg = exporters.save_as_vti(self.xPhys, self.last_params["nelxyz"], filepath)
                if not success:
                    raise Exception(error_msg)

            elif file_type == "stl":
                success, error_msg = exporters.save_as_stl(self.xPhys, self.last_params["nelxyz"], filepath)
                if not success:
                    raise Exception(error_msg)

            self.status_bar.showMessage(f"Result saved to {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the file:\n{e}")
        

    def save_as_png(self):
        """Function connected to the save as PNG button."""
        self.save_result_as('png')

    def save_as_vti(self):
        """Function connected to the save as VTI button."""
        self.save_result_as('vti')

    def save_as_stl(self):
        """Function connected to the save as STL button."""
        self.save_result_as('stl')

    ########
    # PLOT #
    ########

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
        is_3d_mode = self.last_params['nelxyz'][2] > 0
        if is_3d_mode:
            ax = self.figure.add_subplot(111, projection='3d', facecolor='white')
        else:
            ax = self.figure.add_subplot(111, facecolor='white')

        # Layer 1: The Main Result (Material)
        if self.is_displaying_deformation and self.last_displayed_frame_data is not None:
            if self.last_params['disp_iterations'] == 1: # Single-frame grid plot
                if is_3d_mode:
                    nelx, nely, nelz = self.last_params['nelxyz']
                    self.figure.patch.set_facecolor('white')
                    x_phys_3d = self.xPhys.reshape((nelz, nelx, nely)).transpose(1, 2, 0)
                    # Use alpha = density to visualize deformation
                    base_color = np.array(to_rgb(self.material_widget.mat_color.get_color()))
                    colors = np.zeros(x_phys_3d.shape + (4,))
                    colors[..., :3] = base_color
                    colors[..., 3] = np.clip(x_phys_3d, 0.0, 1.0)
                    mask = x_phys_3d > 1e-3 # Only show elements with non-negligible density
                    ax.voxels(mask, facecolors=colors, edgecolor=None)
                    ax.set_box_aspect([nelx, nely, nelz])
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                else:
                    X, Y = self.last_displayed_frame_data
                    nelx, nely = self.last_params['nelxyz'][:2]
                    ax.pcolormesh(X, Y, -self.xPhys.reshape((nelx, nely)), cmap='gray', shading='auto')
            else: # Animation frame density plot
                if is_3d_mode:
                    # TODO
                    i = 1
                else:
                    nx, ny = self.last_params['nelxyz'][:2]
                    mat_color = self.material_widget.mat_color.get_color()
                    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", mat_color])
                    ax.imshow(self.last_displayed_frame_data.reshape((nx, ny)).T, cmap=cmap,
                            interpolation='nearest', origin='lower', norm=plt.Normalize(0, 1))
        else:
            if self.sections['material'].visibility_button.isChecked():
                to_be_initialized = self.xPhys is None
                if self.xPhys is None:
                    p = self.last_params
                    # Initialize xPhys
                    nelx = p['nelxyz'][0]
                    nely = p['nelxyz'][1]
                    nelz = p['nelxyz'][2]
                    active_forces_indices = [i for i in range(len(p['fdir'])) if np.array(p['fdir'])[i] != '-']
                    active_supports_indices = [i for i in range(len(p['sdim'])) if np.array(p['sdim'])[i] != '-']
                    fx_active = np.array(p['fx'])[active_forces_indices]
                    fy_active = np.array(p['fy'])[active_forces_indices]
                    sx_active = np.array(p['sx'])[active_supports_indices]
                    sy_active = np.array(p['sy'])[active_supports_indices]
                    all_x = np.concatenate([fx_active, sx_active])
                    all_y = np.concatenate([fy_active, sy_active])
                    if is_3d_mode:
                        fz_active = np.array(p['fz'])[active_forces_indices]
                        sz_active = np.array(p['sz'])[active_supports_indices]
                    all_z = np.concatenate([fz_active, sz_active]) if is_3d_mode else np.array([0]*len(all_x))
                    self.xPhys = initializers.initialize_material(p['init_type'], p['volfrac'], nelx, nely, nelz, all_x, all_y, all_z)
                    # Add voids if specified
                    for i, shape in enumerate(p['vshape']):
                        if shape == '-': continue
                        x_min, x_max = max(0, int(p['vx'][i] - p['vradius'][i])), min(nelx, int(p['vx'][i] + p['vradius'][i]) + 1)
                        y_min, y_max = max(0, int(p['vy'][i] - p['vradius'][i])), min(nely, int(p['vy'][i] + p['vradius'][i]) + 1)
                        if is_3d_mode: z_min, z_max = max(0, int(p['vz'][i] - p['r'])), min(nelz, int(p['vz'][i] + p['vradius'][i]) + 1)

                        idx_x = np.arange(x_min, x_max)
                        idx_y = np.arange(y_min, y_max)
                        if is_3d_mode: idx_z = np.arange(z_min, z_max)
                        
                        if p['vshape'][i] == '□':  # Square/Cube
                            if len(idx_x) > 0 and len(idx_y) > 0:
                                if (is_3d_mode and len(idx_z) > 0):
                                    xx, yy, zz = np.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
                                    indices = zz + yy * nelz + xx * nely * nelz
                                elif not is_3d_mode:
                                    xx, yy = np.meshgrid(idx_x, idx_y, indexing='ij')
                                    indices = yy + xx * nely

                        elif p['vshape'][i] == '○':  # Circle/Sphere
                            if len(idx_x) > 0 and len(idx_y) > 0:
                                if (is_3d_mode and len(idx_z) > 0):
                                    i_grid, j_grid, k_grid = np.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
                                    mask = (i_grid - p['vx'][i])**2 + (j_grid - p['vy'][i])**2 + (k_grid - p['vz'][i])**2 <= p['vradius'][i]**2
                                    ii, jj, kk = i_grid[mask], j_grid[mask], k_grid[mask]
                                    indices = kk + jj * nelz + ii * nely * nelz
                                elif not is_3d_mode:
                                    i_grid, j_grid = np.meshgrid(idx_x, idx_y, indexing='ij')
                                    mask = (i_grid - p['vx'][i])**2 + (j_grid - p['vy'][i])**2 <= p['vradius'][i]**2
                                    ii, jj = i_grid[mask], j_grid[mask]
                                    indices = jj + ii * nely
                        self.xPhys[indices.flatten()] = 1e-6
                self.plot_material(ax, is_3d=is_3d_mode)
                if to_be_initialized:
                    init_message = 'Configure parameters and press "Create"'
                    if is_3d_mode:
                        ax.text(0.5, 0.5, 0.5, s=init_message, transform=ax.transAxes,
                            ha='center', va='center', fontsize=16, alpha=0.5, color='black')
                    else:
                        ax.text(0.5, 0.5, s=init_message, transform=ax.transAxes,
                            ha='center', va='center', fontsize=16, alpha=0.5, color='black')
        
        self.redraw_non_material_layers(ax, is_3d_mode)
        if not is_3d_mode:
            ax.set_aspect('equal', 'box')
        ax.autoscale(tight=True)
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
        if data_to_plot is None: return
        
        ax.clear()
        if is_3d:
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
            
            # Avoids plotting fully transparent points.
            visible_elements_mask = data_to_plot > 0.01
            visible_indices = np.where(visible_elements_mask)[0]
            
            densities = data_to_plot[visible_indices]

            z = visible_indices // (nx * ny)
            x = (visible_indices % (nx * ny)) // ny
            y = visible_indices % ny
            
            colors = np.zeros((len(densities), 4))
            base_color_rgb = to_rgb(self.material_widget.mat_color.get_color())
            colors[:, :3] = base_color_rgb  # Set the RGB color for all points
            colors[:, 3] = densities # Set the Alpha channel to the density

            ax.scatter(x + 0.5, y + 0.5, z + 0.5,
                       s=6000/max(nx, ny, nz),
                       marker='s', # Square markers to mimic voxels
                       c=colors,
                       alpha=None) # Alpha is now controlled by the 'c' array
            
            ax.set_box_aspect([nx, ny, nz])
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
        self.plot_void_regions(ax, is_3d=is_3d_mode)
        self.plot_dimensions_frame(ax, is_3d=is_3d_mode)
        self.plot_displacement_preview(ax, is_3d=is_3d_mode)

    def plot_dimensions_frame(self, ax, is_3d):
        """Draws a dotted frame around the design space, controlled by the Dimensions section's visibility button."""
        if not self.sections['dimensions'].visibility_button.isChecked():
            ax.set_xlabel(""); ax.set_ylabel("")
            if is_3d: ax.set_zlabel("")
            ax.set_xticks([]); ax.set_yticks([])
            if is_3d: ax.set_zticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
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

        ax.set_xlabel("X", color='black')
        ax.set_ylabel("Y", color='black')
        ax.yaxis.label.set_rotation(0) # Display Y label vertically
        if is_3d: ax.set_zlabel("Z", color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        if is_3d: ax.tick_params(axis='z', colors='black')
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor('black')

    def plot_forces(self, ax, is_3d):
        """Plots the forces as arrows."""
        if not self.sections['forces'].visibility_button.isChecked(): return
        if not self.last_params: return
        p = self.last_params
        if self.is_displaying_deformation and self.u is not None:
            if self.displacement_widget.mov_iter.value() == 1: # Only show forces in single-frame displacement mode, not supported yet in animation mode
                active_forces = [g for g in self.forces_widget.inputs if g['fdir'].currentText() != '-']
                if not active_forces: return
                orig_fx = np.array([g['fx'].value() for g in active_forces])
                orig_fy = np.array([g['fy'].value() for g in active_forces])
                if is_3d: orig_fz = np.array([g['fz'].value() for g in active_forces])
                colors = ['r' if i == 0 else 'b' for i, g in enumerate(self.forces_widget.inputs) if g['fdir'].currentText() != '-']
                disp_factor = self.displacement_widget.mov_disp.value()
                nely = p['nelxyz'][1]
                indices = (orig_fz * (orig_fx + 1) * (nely + 1)) + (orig_fx * (nely + 1)) + orig_fy if is_3d else (orig_fx * (nely + 1)) + orig_fy
                ux = self.u[2 * indices    , 0] * disp_factor
                uy = self.u[2 * indices + 1, 0] * disp_factor
                if is_3d: uz = self.u[2 * indices + 2, 0] * disp_factor
                new_fx = orig_fx + ux
                new_fy = orig_fy + uy if is_3d else orig_fy - uy
                if is_3d: new_fz = orig_fz + uz
                
                length = np.mean(p['nelxyz'][:2]) / 6
                dx, dy = np.zeros_like(new_fx), np.zeros_like(new_fy)
                if is_3d: dz = np.zeros_like(new_fz)
                directions = [g['fdir'].currentText() for g in active_forces]
                for i, d in enumerate(directions):
                    if d == '-': continue
                    elif 'X:→' in d: dx[i] = length
                    elif 'X:←' in d: dx[i] = -length
                    elif 'Y:↑' in d: dy[i] = length
                    elif 'Y:↓' in d: dy[i] = -length
                    elif is_3d:
                        if 'Z:<' in d: dz[i] = length
                        elif 'Z:>' in d: dz[i] = -length
                
                if is_3d:
                    ax.quiver(new_fx, new_fy, new_fz, dx, dy, dz, color=colors, length=length, normalize=True)
                else:
                    ax.quiver(new_fx, new_fy, dx, dy, color=colors, scale_units='xy', angles='xy', scale=1)
        else:
            colors = ['r' if i == 0 else 'b' for i, g in enumerate(self.forces_widget.inputs) if g['fdir'].currentText() != '-']
            dx, dy = np.zeros_like(p['fx']), np.zeros_like(p['fy'])
            if is_3d: dz = np.zeros_like(p['fz'])
            length = np.mean(p['nelxyz'][:2])/6
            directions = p['fdir']
            for i, d in enumerate(directions):
                if d == '-': continue
                elif 'X:→' in d: dx[i] = length
                elif 'X:←' in d: dx[i] = -length
                elif 'Y:↑' in d: dy[i] = length
                elif 'Y:↓' in d: dy[i] = -length
                elif is_3d:
                    if 'Z:<' in d: dz[i] = length
                    elif 'Z:>' in d: dz[i] = -length
            if is_3d:
                ax.quiver(p['fx'], p['fy'], p['fz'], dx, dy, dz, color=colors, length=length, normalize=True)
            else:
                ax.quiver(p['fx'], p['fy'], dx, dy, color=colors, scale_units='xy', angles='xy', scale=1)

    def plot_supports(self, ax, is_3d):
        """Plots the supports as triangles."""
        if not self.sections['supports'].visibility_button.isChecked(): return
        # No need to consider the case is_displaying_deformation since the supports don't move
        p = self.last_params
        for i, d in enumerate(p['sdim']):
            if d == '-': continue
            pos = [p['sx'][i], p['sy'][i], p['sz'][i]]
            if is_3d:
                ax.scatter(pos[0], pos[1], pos[2], s=80, marker='^', c='black', depthshade=False)
            else:
                ax.scatter(pos[0], pos[1], s=80, marker='^', c='black')

    def plot_void_regions(self, ax, is_3d):
        """Plots the void region outline (square/cube or circle/sphere) in 2D or 3D."""
        if not self.sections['void'].visibility_button.isChecked(): return
        if self.is_displaying_deformation: return # Void region are not relevant in deformation view
        
        p = self.last_params
        if not p: return
        for i, d in enumerate(p['vshape']):
            shape = p['vshape'][i]
            if shape == '-': continue

            r = p['vradius'][i]

            if is_3d:
                vx, vy, vz = p['vx'][i], p['vy'][i], p['vz'][i]
                if shape == '□':
                    # Define the 8 vertices of the cube
                    verts = np.array([
                        [vx-r, vy-r, vz-r], [vx+r, vy-r, vz-r], [vx+r, vy+r, vz-r], [vx-r, vy+r, vz-r],
                        [vx-r, vy-r, vz+r], [vx+r, vy-r, vz+r], [vx+r, vy+r, vz+r], [vx-r, vy+r, vz+r]
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

                elif shape == '○':
                    # Create the surface grid for the sphere
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    # Parametric equations for a sphere
                    x = vx + r * np.outer(np.cos(u), np.sin(v))
                    y = vy + r * np.outer(np.sin(u), np.sin(v))
                    z = vz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_wireframe(x, y, z, color='green', linestyle=':')
            
            else:
                vx, vy = p['vx'][i], p['vy'][i]
                if shape == '□':
                    rect = plt.Rectangle((vx-r, vy-r), 2*r, 2*r, fill=False, edgecolor='green', linestyle=':')
                    ax.add_patch(rect)
                elif shape == '○':
                    circ = plt.Circle((vx, vy), r, fill=False, edgecolor='green', linestyle=':')
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

    def plot_displacement_preview(self, ax, is_3d):
        """Overlays displacement vectors (quivers) on the plot if the preview is active."""
        if not self.sections['displacement'].visibility_button.isChecked(): return
        if self.is_displaying_deformation: return # The displacement vector doesn't match the deformed shape
        if self.u is None or self.xPhys is None: return

        p = self.last_params
        disp_factor = self.displacement_widget.mov_disp.value()
        factor = disp_factor / p['fnorm'][0] if p['fnorm'][0] != 0 else disp_factor

        if is_3d:
            nelx, nely, nelz = p['nelxyz']
            step = max(1, int((nelx + nely + nelz) / 15)) # number of elements to skip between 2 arrows
            x_coords, y_coords, z_coords = np.meshgrid(np.arange(0, nelx, step),
                                                       np.arange(0, nely, step),
                                                       np.arange(0, nelz, step), indexing='xy')

            el_indices = (z_coords * (nelx * nely) + x_coords * nely + y_coords).flatten()
            node_indices = (z_coords * ((nelx + 1) * (nely + 1)) + x_coords * (nely + 1) + y_coords).flatten()
            material_mask = self.xPhys[el_indices] > 0.5  # Only show arrows in material regions
            
            # Get the coordinates and displacement vectors for the valid points
            x_valid = x_coords.flatten()[material_mask] + 0.5  # Center of element
            y_valid = y_coords.flatten()[material_mask] + 0.5
            z_valid = z_coords.flatten()[material_mask] + 0.5
            node_valid = node_indices[material_mask]
            
            ux =  self.u[3 * node_valid    , 0] * factor
            uy = -self.u[3 * node_valid + 1, 0] * factor
            uz =  self.u[3 * node_valid + 2, 0] * factor

            ax.quiver(x_valid, y_valid, z_valid, ux, uy, uz, color='red', length=disp_factor/4, normalize=True)
        else:
            nelx, nely = p['nelxyz'][:2]
            step = max(1, int((nelx + nely) / 25)) # number of elements to skip between 2 arrows
            x_coords, y_coords = np.meshgrid(np.arange(0, nelx, step),
                                             np.arange(0, nely, step), indexing='xy')
            
            el_indices = (x_coords * nely + y_coords).flatten()
            node_indices = (x_coords * (nely + 1) + y_coords).flatten()
            
            material_mask = self.xPhys[el_indices] > 0.5 # Only show arrows in material regions
            
            # Get the coordinates and displacement vectors for the valid points
            x_valid = x_coords.flatten()[material_mask]
            y_valid = y_coords.flatten()[material_mask]
            node_valid = node_indices[material_mask]
            
            ux =  self.u[2 * node_valid    , 0] * factor
            uy = -self.u[2 * node_valid + 1, 0] * factor
            
            ax.quiver(x_valid, y_valid, ux, uy, color='red', scale=40, scale_units='xy', angles='xy')

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
        self.block_all_parameter_signals(True)

        # Dimensions
        self.dim_widget.nx.setValue(params['nelxyz'][0])
        self.dim_widget.ny.setValue(params['nelxyz'][1])
        self.dim_widget.nz.setValue(params['nelxyz'][2])
        self.dim_widget.volfrac.setValue(params.get('volfrac', 0.3))
        self.update_position_ranges()
        self.on_mode_changed()
        
        # Void Regions
        for i, void_group in enumerate(self.void_widget.inputs):
            void_group['vshape'].blockSignals(True) # Reblock to avoid triggering on change
            void_group['vshape'].setCurrentText(f"{params['vshape'][0]} (Square)" if params['vshape'][0] == '□' else f"{params['vshape'][0]} (Circle)" if params['vshape'][0] == '○' else '-')
            void_group['vradius'].setValue(params['vradius'][0])
            void_group['vx'].setValue(params['vx'][0])
            void_group['vy'].setValue(params['vy'][0])
            void_group['vz'].setValue(params['vz'][0])
        
        # Forces
        for i, force_group in enumerate(self.forces_widget.inputs):
            force_group['fx'].setValue(params['fx'][i])
            force_group['fy'].setValue(params['fy'][i])
            force_group['fz'].setValue(params['fz'][i])
            force_group['fdir'].setCurrentText(params['fdir'][i])
            force_group['fnorm'].setValue(params['fnorm'][i])
        
        # Supports
        num_supports_in_preset = len(params.get('sx', []))
        for i, support_group in enumerate(self.supports_widget.inputs):
            if i < num_supports_in_preset:
                # If data exists for this support, apply it
                support_group['sx'].setValue(params['sx'][i])
                support_group['sy'].setValue(params['sy'][i])
                support_group['sz'].setValue(params['sz'][i])
                support_group['sdim'].setCurrentText(params['sdim'][i])
            else:
                # If no data exists, reset this row to default empty values
                support_group['sx'].setValue(0)
                support_group['sy'].setValue(0)
                support_group['sz'].setValue(0)
                support_group['sdim'].setCurrentIndex(0) # Set to '-'

        # Material
        self.material_widget.mat_E.setValue(params['E'])
        self.material_widget.mat_nu.setValue(params['nu'])
        self.material_widget.mat_init_type.setCurrentIndex(params['init_type'])

        # Optimizer
        self.optimizer_widget.opt_ft.setCurrentIndex(0 if params['filter_type'] == "Sensitivity" else 1)
        self.optimizer_widget.opt_fr.setValue(params['filter_radius_min'])
        self.optimizer_widget.opt_p.setValue(params['penal'])
        self.optimizer_widget.opt_max_change.setValue(params['max_change'])
        self.optimizer_widget.opt_n_it.setValue(params['n_it'])
        
        # Displacement
        self.displacement_widget.mov_disp.setValue(params['disp_factor'])
        self.displacement_widget.mov_iter.setValue(params['disp_iterations'])
        
        # Unblock signals
        self.block_all_parameter_signals(False)
        
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

        # Update presets icons
        self.header.info_button.setIcon(icons.get('info'))
        self.preset.save_preset_button.setIcon(icons.get('save'))
        self.preset.delete_preset_button.setIcon(icons.get('delete'))
        # Update displacement icons
        self.displacement_widget.run_disp_button.setIcon(icons.get('move'))
        self.displacement_widget.stop_disp_button.setIcon(icons.get('stop'))
        self.displacement_widget.reset_disp_button.setIcon(icons.get('reset'))
        # Update footer icons
        self.footer.create_button.setIcon(icons.get('create'))
        self.footer.stop_button.setIcon(icons.get('Stop'))
        self.footer.save_button.setIcon(icons.get('save'))

        # Update dynamic icons (like the visibility and collapsible arrows)
        for section in self.sections.values():
            section.update_all_icons()

    def open_github_link(self):
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/Topopt-Comec") 
        QDesktopServices.openUrl(url)