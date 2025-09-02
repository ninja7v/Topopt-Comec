# app/ui/widgets.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Custom PySide6 widgets for the TopOpt-Comec UI.

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QSizePolicy,
                               QLabel, QHBoxLayout, QGridLayout, QSpinBox,
                               QDoubleSpinBox, QComboBox, QColorDialog,
                               QGraphicsDropShadowEffect, QToolButton, QMenu, QStackedLayout)
from PySide6.QtGui import QFont, QColor, QAction
from .icons import icons

class CollapsibleSection(QWidget):
    """A collapsible widget section with a title bar and a content area."""
    def __init__(self, title="Section", content_widget=None, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.is_collapsed = True

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Title bar
        self.title_bar = QFrame()
        self.title_bar.setObjectName("collapsibleTitleBar")
        self.title_bar.setLayout(QHBoxLayout())
        self.title_bar.layout().setContentsMargins(5, 2, 5, 2)
        # Expand/Collapse Button
        self.toggle_button = QPushButton()
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setIcon(icons.get('arrow_right'))
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.title_bar.layout().addWidget(self.toggle_button)
        # Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("collapsibleTitleLabel")
        self.title_label.setStyleSheet("background: transparent;")
        self.title_bar.layout().addWidget(self.title_label)
        self.title_bar.layout().addStretch()
        # Visibility Toggle Button
        self.visibility_button = QPushButton()
        self.visibility_button.setIcon(icons.get('eye_open'))
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(True)
        self.visibility_button.setToolTip("Toggle visibility of this element on the plot")
        self.visibility_button.setVisible(False)
        self.title_bar.layout().addWidget(self.visibility_button)

        # Content area
        self.content_widget = content_widget if content_widget else QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        self.content_widget.setObjectName("collapsibleContent")
        self.content_widget.setVisible(not self.is_collapsed)

        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.content_widget)
        
        self.toggle_button.toggled.connect(self.toggle_collapse)
        self.title_bar.mousePressEvent = lambda event: self.toggle_button.toggle()

    def set_visibility_toggle(self, visible: bool):
        self.visibility_button.setVisible(visible)

    def toggle_collapse(self, checked):
        self.is_collapsed = not checked
        self.content_widget.setVisible(not self.is_collapsed)
        if self.is_collapsed:
            self.toggle_button.setIcon(icons.get('arrow_right'))
        else:
            self.toggle_button.setIcon(icons.get('arrow_down'))

    def collapse(self):
        self.toggle_button.setChecked(False)

    def expand(self):
        self.toggle_button.setChecked(True)
    
    def update_all_icons(self):
        """Updates the collapsible arrow and visibility eye icons to match the current theme."""
        # Update the expand/collapse arrow
        if self.is_collapsed:
            self.toggle_button.setIcon(icons.get('arrow_right'))
        else:
            self.toggle_button.setIcon(icons.get('arrow_down'))
        
        # Update the visibility eye icon
        if self.visibility_button.isVisible():
            if self.visibility_button.isChecked():
                self.visibility_button.setIcon(icons.get('eye_open'))
            else:
                self.visibility_button.setIcon(icons.get('eye_closed'))

class ColorPickerButton(QPushButton):
    """A button that opens a color dialog and shows the selected color."""
    def __init__(self, initial_color=QColor("black")):
        super().__init__()
        self.color = initial_color
        self.update_color()
        self.clicked.connect(self.pick_color)
        self.setToolTip("Select material color")

    def pick_color(self):
        new_color = QColorDialog.getColor(self.color, self, "Choose a color")
        if new_color.isValid():
            self.color = new_color
            self.update_color()
    
    def update_color(self):
        self.setStyleSheet(f"background-color: {self.color.name()};")

    def get_color(self):
        return self.color.name()

class HeaderWidget(QWidget):
    """Custom Widget for the control panel's header, including title and action buttons."""
    def __init__(self):
        super().__init__()
        title_layout = QHBoxLayout(self)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addStretch()
        # Title
        title = QLabel("Topopt Comec")
        title_font = QFont("Arial", 20, QFont.Bold)
        title.setFont(title_font)
        title.setToolTip("Topology Optimization for Compliant Mechanisms")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        # Info
        self.info_button = QPushButton()
        self.info_button.setIcon(icons.get('info'))
        self.info_button.setFixedSize(24, 24)
        self.info_button.setToolTip("Open the project's GitHub page")
        self.info_button.setFlat(True)
        title_layout.addWidget(self.info_button)
        title_layout.addStretch()
        # Theme Toggle
        self.theme_button = QPushButton()
        self.theme_button.setIcon(icons.get('sun' if icons.theme == 'dark' else 'moon')) 
        self.theme_button.setToolTip("Switch Theme")
        self.theme_button.setFixedSize(31, 31)
        self.theme_button.setCheckable(True)
        title_layout.addWidget(self.theme_button)

class PresetWidget(QFrame):
    """Custom widget for dimension inputs with a 3D axis visual."""
    def __init__(self):
        super().__init__()
        # Frame
        self.setObjectName("presetFrame")
        preset_layout = QHBoxLayout(self)
        preset_layout.setContentsMargins(5, 5, 5, 5)
        # Combo Box
        preset_layout.addWidget(QLabel("Presets:"))
        self.presets_combo = QComboBox()
        self.presets_combo.setToolTip("Load a saved set of parameters")
        preset_layout.addWidget(self.presets_combo, 1) # Give combo box more stretch
        # Save
        self.save_preset_button = QPushButton()
        self.save_preset_button.setIcon(icons.get('save'))
        self.save_preset_button.setToolTip("Save current parameters as a new preset")
        preset_layout.addWidget(self.save_preset_button)
        # Delete
        self.delete_preset_button = QPushButton()
        self.delete_preset_button.setIcon(icons.get('delete'))
        self.delete_preset_button.setToolTip("Delete the selected preset")
        self.delete_preset_button.setEnabled(False)
        preset_layout.addWidget(self.delete_preset_button)

class DimensionsWidget(QWidget):
    """Custom widget for dimension inputs with a 3D axis visual."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        # Size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.nx = QSpinBox()
        self.nx.setRange(10, 200); self.nx.setValue(60); self.nx.setMaximumWidth(70)
        self.nx.setToolTip("X")
        size_layout.addWidget(self.nx)
        size_layout.addWidget(QLabel("x"))
        self.ny = QSpinBox()
        self.ny.setRange(10, 200); self.ny.setValue(40); self.ny.setMaximumWidth(70)
        self.ny.setToolTip("Y")
        size_layout.addWidget(self.ny)
        size_layout.addWidget(QLabel("x"))
        self.nz = QSpinBox()
        self.nz.setRange(0, 200); self.nz.setValue(0); self.nz.setMaximumWidth(70)
        self.nz.setToolTip("Z → set to 0 for a 2D problem")
        size_layout.addWidget(self.nz)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        # Volume Fraction
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Vol. Frac:"))
        self.volfrac = QDoubleSpinBox()
        self.volfrac.setRange(0.05, 0.8); self.volfrac.setValue(0.3)
        self.volfrac.setSingleStep(0.05)
        self.volfrac.setToolTip("Volume Fraction")
        vol_layout.addWidget(self.volfrac)
        vol_layout.addStretch()
        layout.addLayout(vol_layout)

class VoidWidget(QWidget):
    """Custom widget for void inputs."""
    def __init__(self):
        super().__init__()
        self.inputs = [] # This list will hold the input widgets so the MainWindow can access them
        layout = QGridLayout(self)
        # Shape
        vshape = QComboBox()
        vshape.addItems(['-', '□ (Square)', '○ (Circle)'])
        vshape.setToolTip("Shape of the void")
        shape_radius_layout = QHBoxLayout()
        shape_radius_layout.addWidget(QLabel("Shape:"))
        shape_radius_layout.addWidget(vshape)
        shape_radius_layout.addSpacing(10)
        # Radius
        vradius = QSpinBox()
        vradius.setRange(1, 100); vradius.setMaximumWidth(70)
        vradius.setToolTip("Radius of the shape")
        shape_radius_layout.addWidget(QLabel("Radius:"))
        shape_radius_layout.addWidget(vradius)
        shape_radius_layout.addStretch()
        layout.addLayout(shape_radius_layout, 0, 0, 1, 2)
        # Center
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center:"))
        vx = QSpinBox()
        vx.setRange(0, 200); vx.setMaximumWidth(70)
        vx.setToolTip("X")
        center_layout.addWidget(vx)
        vy = QSpinBox()
        vy.setRange(0, 200); vy.setMaximumWidth(70)
        vy.setToolTip("Y")
        center_layout.addWidget(vy)
        vz = QSpinBox()
        vz.setRange(0, 200); vz.setMaximumWidth(70)
        vz.setToolTip("Z")
        center_layout.addWidget(vz)
        center_layout.addStretch()
        layout.addLayout(center_layout, 1, 0, 1, 2)
        
        self.inputs.append({'vshape': vshape, 'vradius': vradius, 'vx': vx, 'vy': vy, 'vz': vz})
    
    def update_for_mode(self, is_3d: bool):
        """Updates the text in the shape combo box to match the current mode."""
        # Store the current index to re-apply it after changing the items
        for input_group in self.inputs:
            combo_box = input_group['vshape']
            current_index = combo_box.currentIndex()
            
            # Block signals to prevent this change from triggering a replot unnecessarily
            combo_box.blockSignals(True)
            
            # Clear the old items
            combo_box.clear()
            
            if is_3d:
                combo_box.addItems(['-', '□ (Cube)', '○ (Sphere)'])
            else:
                combo_box.addItems(['-', '□ (Square)', '○ (Circle)'])
            
            # Restore the previous selection and unblock signals
            combo_box.setCurrentIndex(current_index)
            combo_box.blockSignals(False)

class ForcesWidget(QWidget):
    """Custom widget for defining the input and output forces."""
    def __init__(self):
        super().__init__()
        self.inputs = [] # This list will hold the input widgets so the MainWindow can access them
        # The main layout for this widget stacks the 3 force sections vertically
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        arrows = ['-', 'X:→', 'X:←', 'Y:↑', 'Y:↓', 'Z:<', 'Z:>']
        force_labels = ["Input", "Output 1", "Output 2"]
        default_pos = [[30, 0, 0], [30, 40, 0], [0, 0, 0]]
        default_arrows = [3, 4, 0]
        default_fv = [0.01, 0.01, 0.0]

        for i, label in enumerate(force_labels):
            # Use a grid to neatly align the rows for this force
            grid = QGridLayout()
            grid.setColumnStretch(1, 1) # Allow input column to expand
            # Force position Row
            force_label = QLabel(label)
            force_label.setStyleSheet("color: red;" if label == "Input" else "color: blue;")
            grid.addWidget(force_label, 0, 0)
            pos_layout = QHBoxLayout()
            fx = QSpinBox()
            fx.setRange(0, 200); fx.setValue(default_pos[i][0]); fx.setMaximumWidth(70)
            fx.setToolTip("X")
            pos_layout.addWidget(fx)
            fy = QSpinBox()
            fy.setRange(0, 200); fy.setValue(default_pos[i][1]); fy.setMaximumWidth(70)
            fy.setToolTip("Y")
            pos_layout.addWidget(fy)
            fz = QSpinBox()
            fz.setRange(0, 200); fz.setValue(default_pos[i][2]); fz.setMaximumWidth(70)
            fz.setToolTip("Z")
            pos_layout.addWidget(fz)
            grid.addLayout(pos_layout, 0, 1)
            # Direction
            dir_layout = QHBoxLayout()
            dir_layout.addWidget(QLabel("Dir:"))
            fdir = QComboBox(); fdir.addItems(arrows); fdir.setCurrentIndex(default_arrows[i])
            fdir.setToolTip("Force direction")
            dir_layout.addWidget(fdir)
            dir_layout.addSpacing(20)
            # Force spring
            dir_layout.addWidget(QLabel("Spring (N/m):"))
            fnorm = QDoubleSpinBox()
            fnorm.setRange(0, 10); fnorm.setSingleStep(0.01); fnorm.setValue(default_fv[i])
            fnorm.setToolTip("Force magnitude for input, spring stiffness for output")
            dir_layout.addWidget(fnorm)
            dir_layout.addStretch()
            grid.addLayout(dir_layout, 1, 0, 1, 2) # Span across both columns

            # Add this force's grid to the main vertical layout
            main_layout.addLayout(grid)
            
            # Store the widgets in the public 'inputs' list
            self.inputs.append({'fx': fx, 'fy': fy, 'fz': fz, 'fdir': fdir, 'fnorm': fnorm})

            # Add a separator line between force sections
            if i < len(force_labels) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                main_layout.addWidget(line)

class SupportWidget(QWidget):
    """Custom widget for defining up to four supports."""
    def __init__(self):
        super().__init__()
        self.inputs = [] # This list will hold the input widgets so the MainWindow can access them
        # The main layout for this widget
        layout = QGridLayout(self)
        layout.setColumnStretch(1, 1) # Allow the position inputs to stretch
        
        dims = ['-', 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ']
        default_pos = [[0, 0, 0], [60, 0, 0], [0, 0, 0], [0, 0, 0]]
        default_dims = [7, 7, 0, 0]

        for i in range(len(default_dims)):
            pos_layout = QHBoxLayout()
            # Force position
            label = QLabel(f"<b>▲ {i+1}</b>")
            layout.addWidget(label, i, 0)
            sx = QSpinBox(); sx.setRange(0, 200); sx.setValue(default_pos[i][0]); sx.setMaximumWidth(65)
            sx.setToolTip("X")
            pos_layout.addWidget(sx)
            sy = QSpinBox(); sy.setRange(0, 200); sy.setValue(default_pos[i][1]); sy.setMaximumWidth(65)
            sy.setToolTip("Y")
            pos_layout.addWidget(sy)
            sz = QSpinBox(); sz.setRange(0, 200); sz.setValue(default_pos[i][2]); sz.setMaximumWidth(65)
            sz.setToolTip("Z")
            pos_layout.addWidget(sz)
            sdim = QComboBox(); sdim.addItems(dims); sdim.setCurrentIndex(default_dims[i])
            # Force direction
            sdim.setToolTip("Fixed direction(s)")
            layout.addLayout(pos_layout, i, 1)
            layout.addWidget(sdim, i, 2)

            # Store the widgets in the public 'inputs' list
            self.inputs.append({'sx': sx, 'sy': sy, 'sz': sz, 'sdim': sdim})

class MaterialWidget(QWidget):
    """Custom widget for material inputs."""
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Young's Modulus
        layout.addWidget(QLabel("Young's Modulus (E):"), 0, 0)
        self.mat_E = QDoubleSpinBox()
        self.mat_E.setRange(0.1, 100); self.mat_E.setValue(1.0)
        self.mat_E.setToolTip("Material’s stiffness")
        layout.addWidget(self.mat_E, 0, 1)
        # Poisson's Ratio
        layout.addWidget(QLabel("Poisson's Ratio (ν):"), 1, 0)
        self.mat_nu = QDoubleSpinBox()
        self.mat_nu.setRange(0.0, 0.5); self.mat_nu.setValue(0.25); self.mat_nu.setSingleStep(0.05)
        self.mat_nu.setToolTip("Material’s lateral shrinkage relative to its elongation")
        layout.addWidget(self.mat_nu, 1, 1)
        # Color
        layout.addWidget(QLabel("Color:"), 2, 0)
        self.mat_color = ColorPickerButton()
        self.mat_E.setToolTip("Color used in the plot for this material")
        layout.addWidget(self.mat_color, 2, 1)

class OptimizerWidget(QWidget):
    """Custom widget for optimizer inputs."""
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Filter type
        self.opt_ft = QComboBox(); self.opt_ft.addItems(['Sensitivity', 'Density']); self.opt_ft.setCurrentIndex(0)
        self.opt_ft.setToolTip("Sensitivity filter: Averages sensitivities to ensure stable and physically meaningful optimization gradients.\n"
                               "Density filter: Smooths the material distribution to avoid checkerboard patterns and mesh dependency.")
        layout.addWidget(QLabel("Filter Type:"), 0, 0)
        layout.addWidget(self.opt_ft, 0, 1)
        # Filter Radius
        layout.addWidget(QLabel("Filter Radius:"), 1, 0)
        self.opt_fr = QDoubleSpinBox()
        self.opt_fr.setRange(0.1, 10.0); self.opt_fr.setValue(1.3)
        self.opt_fr.setToolTip("Range of the filter coverage")
        layout.addWidget(self.opt_fr, 1, 1)
        # Penalization
        layout.addWidget(QLabel("Penalization:"), 2, 0)
        self.opt_p = QDoubleSpinBox()
        self.opt_p.setRange(1.0, 10.0); self.opt_p.setValue(3.0)
        self.opt_p.setToolTip("Exponent in the SIMP method to penalize intermediate densities")
        layout.addWidget(self.opt_p, 2, 1)
        # Max change
        layout.addWidget(QLabel("Max change:"), 3, 0)
        self.opt_max_change = QDoubleSpinBox()
        self.opt_max_change.setRange(0.01, 0.5); self.opt_max_change.setValue(0.05); self.opt_max_change.setSingleStep(0.05)
        self.opt_max_change.setToolTip("Bound the density change between two iterations to a maximum value")
        layout.addWidget(self.opt_max_change, 3, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 4, 0)
        self.opt_n_it = QSpinBox()
        self.opt_n_it.setRange(1, 100); self.opt_n_it.setValue(30)
        self.opt_n_it.setToolTip("Number of optimization iterations")
        layout.addWidget(self.opt_n_it, 4, 1)

class DisplacementWidget(QWidget):
    """Custom widget for displacement inputs."""
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Displacement
        layout.addWidget(QLabel("Displacement:"), 0, 0)
        self.mov_disp = QDoubleSpinBox()
        self.mov_disp.setRange(0.1, 100.0); self.mov_disp.setValue(1.0)
        self.mov_disp.setToolTip("Total scaling factor for the displacement animation")
        layout.addWidget(self.mov_disp, 0, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.mov_iter = QSpinBox()
        self.mov_iter.setRange(1, 20); self.mov_iter.setValue(1)
        self.mov_iter.setToolTip("Number of frames in the displacement animation.\n"
                                 "If more than 1 iteration is set, a more complex function is used since it requires an domain enlargement and interpolations.")
        layout.addWidget(self.mov_iter, 1, 1)
        self.button_stack = QStackedLayout()
        # Move button
        self.run_disp_button = QPushButton("Move")
        self.run_disp_button.setIcon(icons.get('move'))
        self.run_disp_button.setToolTip("Start the displacement process")
        self.button_stack.addWidget(self.run_disp_button)
        # Stop button
        self.stop_disp_button = QPushButton(" Stop")
        self.stop_disp_button.setIcon(icons.get('stop'))
        self.stop_disp_button.setToolTip("Stop the displacement process")
        self.stop_disp_button.setStyleSheet("background-color: #C0392B;")
        self.button_stack.addWidget(self.stop_disp_button)
        # Reset button
        self.reset_disp_button = QPushButton("Reset View")
        self.reset_disp_button.setIcon(icons.get('reset'))
        self.reset_disp_button.setToolTip("Reset the mechasim to its original position")
        self.button_stack.addWidget(self.reset_disp_button)
        self.button_stack_widget = QWidget()
        self.button_stack_widget.setLayout(self.button_stack)
        layout.addWidget(self.button_stack_widget, 2, 0, 1, 2)

class FooterWidget(QWidget):
    """Custom widget for the footer."""
    def __init__(self):
        super().__init__()
        # Create button
        action_layout = QHBoxLayout(self)
        self.create_button = QPushButton(" Create")
        self.create_button.setObjectName("create_button")
        self.create_button.setIcon(icons.get('create'))
        self.create_button.setToolTip("Start the optimization process")
        self.create_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_create_button_effect()
        action_layout.addWidget(self.create_button)
        # Stop button
        self.stop_button = QPushButton(" Stop")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setIcon(icons.get('stop'))
        self.stop_button.setToolTip("Stop the optimization process")
        self.stop_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #C0392B;")
        self.stop_button.hide() # Hidden by default
        action_layout.addWidget(self.stop_button)
        # Binarize button
        self.binarize_button = QPushButton()
        self.binarize_button.setIcon(icons.get('binarize'))
        self.binarize_button.setToolTip("Apply threshold to make all colors solid (0 or 1)")
        self.binarize_button.setFixedSize(29, 29)
        self.binarize_button.setEnabled(False)
        action_layout.addWidget(self.binarize_button)
        # Save button
        self.save_button = QToolButton()
        self.save_button.setIcon(icons.get('save'))
        self.save_button.setToolTip("Save the current result")
        self.save_button.setFixedSize(30, 30) # A bit wider to accommodate the arrow
        self.save_button.setEnabled(False)
        self.save_button.setPopupMode(QToolButton.InstantPopup)

        save_menu = QMenu(self.save_button)
        
        self.save_png_action = QAction("Save as PNG Image...", self)
        save_menu.addAction(self.save_png_action)
        self.save_vti_action = QAction("Save as VTI (for ParaView)...", self)
        save_menu.addAction(self.save_vti_action)
        self.save_stl_action = QAction("Save as STL (for CAD)...", self)
        save_menu.addAction(self.save_stl_action)

        self.save_button.setMenu(save_menu)

        action_layout.addWidget(self.save_button)

    def start_create_button_effect(self):
        # Shadow effect for glow
        create_button_effect = QGraphicsDropShadowEffect(self.create_button)
        create_button_effect.setBlurRadius(20)
        create_button_effect.setOffset(0)
        create_button_effect.setColor(QColor("#F97316"))
        self.create_button.setGraphicsEffect(create_button_effect)
        # Animation for the shadow
        anim = QPropertyAnimation(create_button_effect, b"blurRadius", self)
        anim.setStartValue(5)
        anim.setEndValue(25)
        anim.setDuration(2000)
        anim.setEasingCurve(QEasingCurve.Type.CosineCurve)
        anim.setLoopCount(-1)
        anim.start()
    
    def stop_create_button_effect(self):
        self.create_button.setGraphicsEffect(None)