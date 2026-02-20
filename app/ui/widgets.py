# app/ui/widgets.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Custom PySide6 widgets for the TopOpt-Comec UI.

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import icons


# Make spinbox functions to avoid code repetition
def make_spin(min_v, max_v, val, width=None, tip=""):
    s = QSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    if width:
        s.setMaximumWidth(width)
    s.setToolTip(tip)
    return s


def make_dspin(min_v, max_v, val, step=0.01, width=None, tip=""):
    s = QDoubleSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    s.setSingleStep(step)
    if width:
        s.setMaximumWidth(width)
    s.setToolTip(tip)
    return s


def make_combo(items=[], index=0, tip=""):
    c = QComboBox()
    c.addItems(items)
    c.setCurrentIndex(index)
    c.setToolTip(tip)
    return c


class CollapsibleSection(QWidget):
    """A collapsible widget section with a title bar and a content area."""

    def __init__(self, title="Section", content_widget=None, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
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
        self.toggle_button.setFixedSize(18, 18)
        self.toggle_button.setIcon(icons.get("arrow_right"))
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
        self.visibility_button.setIcon(icons.get("eye_open"))
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(True)
        self.visibility_button.setToolTip(
            "Toggle visibility of this element on the plot"
        )
        self.visibility_button.setVisible(False)
        self.visibility_button.setFixedSize(25, 18)
        self.title_bar.layout().addWidget(self.visibility_button)

        # Content area
        self.content_widget = content_widget if content_widget else QWidget()
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        self.content_widget.setObjectName("collapsibleContent")
        self.content_widget.setVisible(not self.is_collapsed)

        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.content_widget)

        self.title_bar.setFixedHeight(25)

        self.toggle_button.toggled.connect(self.toggle_collapse)
        self.title_bar.mousePressEvent = lambda event: self.toggle_button.toggle()

    def set_visibility_toggle(self, visible: bool):
        self.visibility_button.setVisible(visible)

    def toggle_collapse(self, checked):
        self.is_collapsed = not checked
        self.content_widget.setVisible(not self.is_collapsed)
        if self.is_collapsed:
            self.toggle_button.setIcon(icons.get("arrow_right"))
        else:
            self.toggle_button.setIcon(icons.get("arrow_down"))

    def collapse(self):
        self.toggle_button.setChecked(False)

    def expand(self):
        self.toggle_button.setChecked(True)

    def update_all_icons(self):
        """Updates the collapsible arrow and visibility eye icons to match the current theme."""
        # Update the expand/collapse arrow
        if self.is_collapsed:
            self.toggle_button.setIcon(icons.get("arrow_right"))
        else:
            self.toggle_button.setIcon(icons.get("arrow_down"))

        # Update the visibility eye icon
        if self.visibility_button.isVisible():
            if self.visibility_button.isChecked():
                self.visibility_button.setIcon(icons.get("eye_open"))
            else:
                self.visibility_button.setIcon(icons.get("eye_closed"))


class ColorPickerButton(QPushButton):
    """A button that opens a color dialog and shows the selected color."""

    colorChanged = Signal(QColor)

    def __init__(self, initial_color=QColor("black")):
        super().__init__()
        self.setObjectName("ColorPickerButton")
        if isinstance(initial_color, str):
            initial_color = QColor(initial_color)

        self.color = initial_color
        self.update_color()
        self.clicked.connect(self.pick_color)
        self.setToolTip("Select material color")

    def pick_color(self):
        new_color = QColorDialog.getColor(self.color, self, "Choose a color")
        if new_color.isValid():
            self.color = new_color
            self.update_color()
            self.colorChanged.emit(self.color)

    def update_color(self):
        self.setStyleSheet(
            f"QPushButton#ColorPickerButton {{ background-color: {self.color.name()}; }}"
        )

    def get_color(self):
        return self.color.name()

    def set_color(self, color: str | QColor):
        if isinstance(color, str):
            color = QColor(color)

        if color != self.color:
            self.color = color
            self.update_color()
            self.colorChanged.emit(color)


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
        self.info_button.setIcon(icons.get("info"))
        self.info_button.setFixedSize(24, 24)
        self.info_button.setToolTip("Open the project's GitHub page")
        self.info_button.setFlat(True)
        title_layout.addWidget(self.info_button)
        title_layout.addStretch()
        # Theme Toggle
        self.theme_button = QPushButton()
        self.theme_button.setIcon(icons.get("sun" if icons.theme == "dark" else "moon"))
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
        self.presets_combo = make_combo([], 0, "Load a saved set of parameters")
        preset_layout.addWidget(self.presets_combo, 1)  # Give combo box more stretch
        # Save
        self.save_preset_button = QPushButton()
        self.save_preset_button.setIcon(icons.get("save"))
        self.save_preset_button.setToolTip("Save current parameters as a new preset")
        preset_layout.addWidget(self.save_preset_button)
        # Delete
        self.delete_preset_button = QPushButton()
        self.delete_preset_button.setIcon(icons.get("delete"))
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
        self.nx = make_spin(5, 1000, 60, 70, "X")
        size_layout.addWidget(self.nx)
        size_layout.addWidget(QLabel("x"))
        self.ny = make_spin(5, 1000, 40, 70, "Y")
        size_layout.addWidget(self.ny)
        size_layout.addWidget(QLabel("x"))
        self.nz = make_spin(0, 1000, 0, 70, "Z → set to 0 for a 2D problem")
        size_layout.addWidget(self.nz)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        # Volume Fraction
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Vol. Frac:"))
        self.volfrac = make_dspin(0.05, 0.8, 0.3, 0.05, None, "Volume Fraction")
        vol_layout.addWidget(self.volfrac)
        vol_layout.addStretch()
        layout.addLayout(vol_layout)
        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale = make_dspin(
            0.5, 5, 1.0, 0.5, None, "Scale every component by this factor"
        )
        scale_layout.addWidget(self.scale)
        self.scale_button = QPushButton("Scale")
        self.scale_button.setIcon(icons.get("scale"))
        self.scale_button.setToolTip("Scale elements")
        scale_layout.addWidget(self.scale_button)
        scale_layout.addStretch(1)
        layout.addLayout(scale_layout)


class RegionsWidget(QWidget):
    """Custom widget for regions inputs."""

    nbRegionsChanged = Signal()

    def __init__(self):
        super().__init__()
        self.inputs = (
            []
        )  # This list will hold the input widgets so the MainWindow can access them

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Layout to hold the region containers
        self.regions_layout = QVBoxLayout()
        self.regions_layout.setSpacing(10)
        self.main_layout.addLayout(self.regions_layout)

        # Add button
        self.add_btn = QPushButton("+ Add Region")
        self.add_btn.clicked.connect(lambda: self.add_region())
        self.add_btn.setToolTip("Add a region")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)
        self.main_layout.addStretch()

    def add_region(
        self, rshape="□", rstate="Void", rradius=5, pos=None, emit_signal=True
    ):
        row = len(self.inputs)
        if row >= 10:  # safety
            return

        if pos is None:
            pos = [0, 0, 0]

        # Container Widget for this region
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # --- Line 1: [Minus] "Shape" [Combos] "Radius" [Spin] ---
        line1_layout = QHBoxLayout()

        # Remove button
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this region")
        remove_btn.clicked.connect(lambda: self.remove_region_by_widget(container))
        line1_layout.addWidget(remove_btn)

        # Shape
        line1_layout.addWidget(QLabel("Shape:"))
        rshape = make_combo(["-", "□", "◯"], 0, "Shape of the region")
        line1_layout.addWidget(rshape)
        rstate = make_combo(["Void", "Filled"], 0, "State of the region")
        line1_layout.addWidget(rstate)
        # Radius
        line1_layout.addWidget(QLabel("Radius:"))
        rradius = make_spin(1, 100, rradius, 60, "Radius")
        line1_layout.addWidget(rradius)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # --- Line 2: "Center" [X] [Y] [Z] ---
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)  # Align with line above
        line2_layout.addWidget(QLabel("Center:"))
        rx = make_spin(0, 1000, pos[0], 60, "X")
        line2_layout.addWidget(rx)
        ry = make_spin(0, 1000, pos[1], 60, "Y")
        line2_layout.addWidget(ry)
        rz = make_spin(0, 1000, pos[2], 60, "Z")
        line2_layout.addWidget(rz)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        # Add to main layout (before the Add button)
        # self.main_layout has: grid (now empty/unused?), add_btn
        # We should insert the container into a specific layout for regions.
        # To avoid index mess, let's create a 'regions_layout' VBox
        if not hasattr(self, "regions_layout"):
            pass
        self.regions_layout.addWidget(container)

        self.inputs.append(
            {
                "rshape": rshape,
                "rstate": rstate,
                "rradius": rradius,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "remove_btn": remove_btn,
                "container": container,
            }
        )

        if emit_signal:
            self.nbRegionsChanged.emit()
        self.update_remove_buttons()

    def remove_region_by_widget(self, container_widget):
        # Find index
        idx = -1
        for i, region in enumerate(self.inputs):
            if region["container"] == container_widget:
                idx = i
                break

        if idx != -1:
            self.remove_region(idx)

    def remove_region(self, row, emit_signal=True):
        if row < 0 or row >= len(self.inputs):
            return

        # Remove widgets
        region = self.inputs.pop(row)
        # We just need to delete the container
        region["container"].deleteLater()
        region["container"].setParent(None)

        # No need to rebuild grid!
        if emit_signal:
            self.nbRegionsChanged.emit()
        self.update_remove_buttons()

    def rebuild_grid(self):
        pass  # No longer needed, but keeping for compatibility if tests call it?
        # Actually tests don't call it directly.

    def update_remove_buttons(self):
        enabled = len(self.inputs) > 0
        for region in self.inputs:
            region["remove_btn"].setEnabled(enabled)


class ForcesWidget(QWidget):
    """Custom widget for defining the input and output forces."""

    def __init__(self):
        super().__init__()
        self.inputs = (
            []
        )  # This list will hold the input widgets so the MainWindow can access them

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        arrows = ["-", "X:→", "X:←", "Y:↑", "Y:↓", "Z:<", "Z:>"]
        iorigins, iarrows, inorms = [[30, 0, 0], [0, 0, 0]], [3, 0], [0.01, 0.0]
        oorigins, oarrows, onorms = [[30, 40, 0], [0, 0, 0]], [4, 0], [0.01, 0.0]

        self._add_section("Input", "red", 2, iorigins, arrows, iarrows, inorms, True)
        self._add_section("Output", "blue", 2, oorigins, arrows, oarrows, onorms, False)

    def _add_section(
        self, title, color, count, origins, arrows, arrow_indices, norms, is_input
    ):
        label = QLabel(title)
        label.setStyleSheet(f"color: {color};")
        self.main_layout.addWidget(label)
        for i in range(count):
            self._add_row(i, origins, arrows, arrow_indices, norms, is_input)
            if i < count - 1:
                self._add_separator()

    def _add_row(self, i, origins, arrows, arrow_indices, norms, is_input):
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Origin:"), 0, 0)

        pos_layout = QHBoxLayout()
        key_prefix = "fi" if is_input else "fo"

        fx = make_spin(0, 1000, origins[i][0], 70, "X")
        fy = make_spin(0, 1000, origins[i][1], 70, "Y")
        fz = make_spin(0, 1000, origins[i][2], 70, "Z")
        for w in [fx, fy, fz]:
            pos_layout.addWidget(w)
        grid.addLayout(pos_layout, 0, 1)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Dir:"))
        fdir = make_combo(arrows, arrow_indices[i], "Force direction")
        dir_layout.addWidget(fdir)
        dir_layout.addSpacing(20)
        dir_layout.addWidget(QLabel("Spring (N/m):"))
        fnorm = make_dspin(
            0,
            10,
            norms[i],
            0.01,
            None,
            "Force magnitude for input, spring stiffness for output",
        )
        dir_layout.addWidget(fnorm)
        dir_layout.addStretch()
        grid.addLayout(dir_layout, 1, 0, 1, 2)

        self.main_layout.addLayout(grid)
        self.inputs.append(
            {
                f"{key_prefix}x": fx,
                f"{key_prefix}y": fy,
                f"{key_prefix}z": fz,
                f"{key_prefix}dir": fdir,
                f"{key_prefix}norm": fnorm,
            }
        )

    def _add_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)


class SupportWidget(QWidget):
    """Custom widget for defining up to four supports."""

    nbSupportsChanged = (
        Signal()
    )  # Signal to update the parameters when the number of supports changes

    def __init__(self):
        super().__init__()
        self.inputs = (
            []
        )  # This list will hold the input widgets so the MainWindow can access them

        self.main_layout = QVBoxLayout(self)

        # Container for the list of supports to ensure vertical stacking
        self.supports_container = QWidget()
        self.supports_list_layout = QVBoxLayout(self.supports_container)
        self.supports_list_layout.setContentsMargins(0, 0, 0, 0)
        self.supports_list_layout.setSpacing(10)
        self.main_layout.addWidget(self.supports_container)

        # Add button
        self.add_btn = QPushButton("+ Add Support")
        self.add_btn.clicked.connect(lambda: self.add_support())
        self.add_btn.setToolTip("Add a support")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)

        self.dims = ["-", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]

    def add_support(self, pos=None, dim="XYZ", radius=0, emit_signal=True):
        row = len(self.inputs)
        if row >= 10:  # safety
            return

        if pos is None:
            pos = [0, 0, 0]

        # Container Widget for this support
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # --- Line 1: "Center:" [X] [Y] [Z] ---
        line1_layout = QHBoxLayout()

        # Remove button
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this support")
        remove_btn.clicked.connect(lambda: self.remove_support_by_widget(container))
        line1_layout.addWidget(remove_btn)

        line1_layout.addWidget(QLabel("Center:"))
        sx = make_spin(0, 1000, pos[0], 65, "X")
        sy = make_spin(0, 1000, pos[1], 65, "Y")
        sz = make_spin(0, 1000, pos[2], 65, "Z")
        line1_layout.addWidget(sx)
        line1_layout.addWidget(sy)
        line1_layout.addWidget(sz)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # --- Line 2: "Fixed:" [Dim] "Radius:" [Radius] ---
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)  # Align with line above
        line2_layout.addWidget(QLabel("Fixed:"))
        sdim = make_combo(self.dims, self.dims.index(dim), "Fixed direction(s)")
        line2_layout.addWidget(sdim)

        line2_layout.addWidget(QLabel("Radius:"))
        sr = make_spin(0, 10, radius, 60, "Support's radius (0 = single node)")
        line2_layout.addWidget(sr)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        self.supports_list_layout.addWidget(container)

        # Store the widgets in the public 'inputs' list
        data = {
            "sx": sx,
            "sy": sy,
            "sz": sz,
            "sdim": sdim,
            "sr": sr,
            "remove_btn": remove_btn,
            "container": container,
        }
        self.inputs.append(data)
        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def remove_support_by_widget(self, container_widget):
        # Find index
        idx = -1
        for i, support in enumerate(self.inputs):
            if support["container"] == container_widget:
                idx = i
                break

        if idx != -1:
            self.remove_support(idx)

    def remove_support(self, row, emit_signal=True):
        if row < 0 or row >= len(self.inputs):
            return

        # Remove widgets
        support = self.inputs.pop(row)
        support["container"].deleteLater()
        support["container"].setParent(None)

        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def update_remove_buttons(self):
        enabled = len(self.inputs) > 0
        for support in self.inputs:
            support["remove_btn"].setEnabled(enabled)


class MaterialsWidget(QWidget):
    """Custom widget for defining the materials."""

    nbMaterialsChanged = (
        Signal()
    )  # Signal to update the parameters when the number of material changes

    def __init__(self):
        super().__init__()
        self.inputs = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Initialization Type (Global for all materials)
        init_layout = QHBoxLayout()
        init_layout.addWidget(QLabel("Initialization:"))
        self.mat_init_type = make_combo(
            ["Uniform", "Around activity points", "Random"],
            0,
            "Material distribution initialization type",
        )
        init_layout.addWidget(self.mat_init_type)
        self.main_layout.addLayout(init_layout)

        # Layout for material items
        self.materials_layout = QVBoxLayout()
        self.materials_layout.setSpacing(10)
        self.main_layout.addLayout(self.materials_layout)

        # Add button
        self.add_btn = QPushButton("+ Add Material")
        self.add_btn.clicked.connect(lambda: self.add_material())
        self.add_btn.setToolTip("Add a material (max 2)")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)

        self.main_layout.addStretch()

        # Add default material
        self.add_material(E=1.0, nu=0.3, percent=100, color="#000000")

    def add_material(self, E=1.0, nu=0.3, percent=0, color="#000000", emit_signal=True):
        if len(self.inputs) > 1:
            return

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)

        # Line 1: [Minus] Color [Btn] % [Spin]
        line1 = QHBoxLayout()
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this material")
        remove_btn.clicked.connect(lambda: self.remove_material_by_widget(container))
        line1.addWidget(remove_btn)
        # Color
        line1.addWidget(QLabel("Color:"))
        if isinstance(color, str):
            color = QColor(color)
        color_btn = ColorPickerButton(color)
        line1.addWidget(color_btn)
        line1.addSpacing(30)
        # Percentage
        line1.addWidget(QLabel("%:"))
        mat_percent = make_spin(0, 100, percent, 70, "Volume Fraction percentage")
        mat_percent.setSingleStep(5)
        line1.addWidget(mat_percent)
        line1.addStretch()
        layout.addLayout(line1)

        # Line 2: E [Spin] nu [Spin]
        line2 = QHBoxLayout()
        line2.addSpacing(40)  # Align with line above
        # Young's Modulus
        line2.addWidget(QLabel("E:"))
        mat_E = make_dspin(
            0.1, 100.0, E, 0.01, 70, "Young's Modulus, material’s stiffness"
        )
        line2.addWidget(mat_E)
        line2.addSpacing(20)
        # Poisson's Ratio
        line2.addWidget(QLabel("ν:"))
        mat_nu = make_dspin(
            0.0,
            0.49,
            nu,
            0.05,
            70,
            "Poisson's Ratio, material’s lateral shrinkage relative to its elongation",
        )
        line2.addWidget(mat_nu)
        line2.addStretch()
        layout.addLayout(line2)

        self.materials_layout.addWidget(container)

        self.inputs.append(
            {
                "E": mat_E,
                "nu": mat_nu,
                "percent": mat_percent,
                "color": color_btn,
                "remove_btn": remove_btn,
                "container": container,
            }
        )

        if emit_signal:
            self.nbMaterialsChanged.emit()
        self.update_ui_state()

    def remove_material_by_widget(self, container):
        idx = -1
        for i, mat in enumerate(self.inputs):
            if mat["container"] == container:
                idx = i
                break
        if idx != -1:
            self.remove_material(idx)

    def remove_material(self, row, emit_signal=True):
        if row < 0 or row >= len(self.inputs):
            return

        mat = self.inputs.pop(row)
        mat["container"].deleteLater()
        mat["container"].setParent(None)

        if emit_signal:
            self.nbMaterialsChanged.emit()
        self.update_ui_state()

    def update_ui_state(self):
        # Update Add/Remove buttons state
        can_add = len(self.inputs) < 2
        can_remove = len(self.inputs) > 1

        self.add_btn.setEnabled(can_add)
        for mat in self.inputs:
            mat["remove_btn"].setEnabled(can_remove)


class OptimizerWidget(QWidget):
    """Custom widget for optimizer inputs."""

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Filter type
        tip = (
            "Regularization to avoid checkerboards and mesh dependency\n"
            "Sensitivity: Averages sensitivities to ensure stable and physically meaningful optimization gradients.\n"
            "Density: Smooths the material distribution to avoid checkerboard patterns and mesh dependency."
        )
        self.opt_ft = make_combo(["Sensitivity", "Density", "None"], 0, tip)
        layout.addWidget(QLabel("Filter Type:"), 0, 0)
        layout.addWidget(self.opt_ft, 0, 1)
        # Filter Radius
        layout.addWidget(QLabel("Filter Radius:"), 1, 0)
        self.opt_fr = make_dspin(
            0.1, 10.0, 1.3, 0.01, None, "Range of the filter coverage"
        )
        layout.addWidget(self.opt_fr, 1, 1)
        # Penalization
        layout.addWidget(QLabel("Penalization:"), 2, 0)
        self.opt_p = make_dspin(
            1.0,
            10.0,
            3.0,
            0.01,
            None,
            "Exponent in the SIMP method to penalize intermediate densities",
        )
        layout.addWidget(self.opt_p, 2, 1)
        # Eta
        layout.addWidget(QLabel("Eta:"), 3, 0)
        self.opt_eta = make_dspin(
            0.05,
            1.0,
            0.3,
            0.05,
            None,
            "Damping factor in OC update rule to controls aggressiveness of density updates.\nLower eta: slower, more stable convergence; Higher eta: faster, but risk",
        )
        layout.addWidget(self.opt_eta, 3, 1)
        # Max change
        layout.addWidget(QLabel("Max change:"), 4, 0)
        self.opt_max_change = make_dspin(
            0.01,
            0.5,
            0.05,
            0.05,
            None,
            "Bound the density change between two iterations to a maximum value",
        )
        layout.addWidget(self.opt_max_change, 4, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 5, 0)
        self.opt_n_it = make_spin(1, 100, 30, None, "Number of optimization iterations")
        layout.addWidget(self.opt_n_it, 5, 1)
        # Solver
        layout.addWidget(QLabel("Solver:"), 6, 0)
        tip = (
            "Solver type for the linear system\n"
            "Auto: Chooses the best solver based on the problem size\n"
            "Direct: Uses LU factorization (spsolve)\n"
            "Iterative: Uses Conjugate Gradient (cg) with preconditioning\n"
        )
        self.opt_solver = make_combo(["Auto", "Direct", "Iterative"], 0, tip)
        layout.addWidget(self.opt_solver, 6, 1)


class AnalysisWidget(QWidget):
    """Custom widget for optimizer inputs."""

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Checkerboard
        layout.addWidget(QLabel("Checkerboard:"), 0, 0)
        self.checkerboard_result = QLabel("-")
        self.checkerboard_result.setToolTip(
            "Check if the mechanism contains some checkerboard paterns"
        )
        layout.addWidget(self.checkerboard_result, 0, 1)
        # Watertight
        layout.addWidget(QLabel("Watertight:"), 1, 0)
        self.watertight_result = QLabel("-")
        self.watertight_result.setToolTip("Check if the mechanism is watertight")
        layout.addWidget(self.watertight_result, 1, 1)
        # Threshold
        layout.addWidget(QLabel("Thresholded:"), 2, 0)
        self.threshold_result = QLabel("-")
        self.threshold_result.setToolTip("Evaluate if gray zones are gone")
        layout.addWidget(self.threshold_result, 2, 1)
        # Efficiency
        layout.addWidget(QLabel("Efficient:"), 3, 0)
        self.efficiency_result = QLabel("-")
        self.efficiency_result.setToolTip(
            "Compare output displacement with input displacement"
        )
        layout.addWidget(self.efficiency_result, 3, 1)
        self.button_stack = QStackedLayout()
        # Analyze button
        self.run_analysis_button = QPushButton("🔍 Analyze")
        self.run_analysis_button.setToolTip("Start the analysis process")
        self.button_stack.addWidget(self.run_analysis_button)
        # Stop button
        self.stop_analysis_button = QPushButton(" Stop")
        self.stop_analysis_button.setObjectName("stop_analysis_button")
        self.stop_analysis_button.setIcon(icons.get("stop"))
        self.stop_analysis_button.setToolTip("Stop the analysis process")
        self.stop_analysis_button.setStyleSheet("background-color: #C0392B;")
        self.button_stack.addWidget(self.stop_analysis_button)
        self.button_stack_widget = QWidget()
        self.button_stack_widget.setLayout(self.button_stack)
        layout.addWidget(self.button_stack_widget, 4, 0, 1, 2)

        ## Analyze button
        # self.run_analysis_button = QPushButton("🔍 Analyze")
        # self.run_analysis_button.setToolTip("Start the analysis process")
        # layout.addWidget(self.run_analysis_button)
        ## Stop button
        # self.stop_analysis_button = QPushButton(" Stop")
        # self.stop_analysis_button.setObjectName("stop_analysis_button")
        # self.stop_analysis_button.setIcon(icons.get("stop"))
        # self.stop_analysis_button.setToolTip("Stop the analysis process")
        # self.stop_analysis_button.setStyleSheet("background-color: #C0392B;")
        # self.stop_analysis_button.hide()  # Hidden by default
        # layout.addWidget(self.stop_analysis_button)


class DisplacementWidget(QWidget):
    """Custom widget for displacement inputs."""

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        # Displacement
        layout.addWidget(QLabel("Displacement:"), 0, 0)
        self.mov_disp = make_dspin(
            0.1,
            100.0,
            1.0,
            0.01,
            None,
            "Total scaling factor for the displacement animation",
        )
        layout.addWidget(self.mov_disp, 0, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.mov_iter = make_spin(
            1,
            20,
            1,
            None,
            "Number of frames in the displacement animation.\nIf more than 1 iteration is set, a more complex function is used since it requires an domain enlargement and interpolations.",
        )
        layout.addWidget(self.mov_iter, 1, 1)
        self.button_stack = QStackedLayout()
        # Move button
        self.run_disp_button = QPushButton("Move")
        self.run_disp_button.setIcon(icons.get("move"))
        self.run_disp_button.setToolTip("Start the displacement process")
        self.button_stack.addWidget(self.run_disp_button)
        # Stop button
        self.stop_disp_button = QPushButton(" Stop")
        self.stop_disp_button.setIcon(icons.get("stop"))
        self.stop_disp_button.setToolTip("Stop the displacement process")
        self.stop_disp_button.setStyleSheet("background-color: #C0392B;")
        self.button_stack.addWidget(self.stop_disp_button)
        # Reset button
        self.reset_disp_button = QPushButton("Reset View")
        self.reset_disp_button.setIcon(icons.get("reset"))
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
        self.create_button.setIcon(icons.get("create"))
        self.create_button.setToolTip("Start the optimization process")
        self.create_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_create_button_effect()
        action_layout.addWidget(self.create_button)
        # Stop button
        self.stop_button = QPushButton(" Stop")
        self.stop_button.setIcon(icons.get("stop"))
        self.stop_button.setToolTip("Stop the optimization process")
        self.stop_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #C0392B;")
        self.stop_button.hide()  # Hidden by default
        action_layout.addWidget(self.stop_button)
        # Binarize button
        self.binarize_button = QPushButton()
        self.binarize_button.setIcon(icons.get("binarize"))
        self.binarize_button.setToolTip(
            "Apply threshold to make all colors solid (0 or 1)"
        )
        self.binarize_button.setFixedSize(29, 29)
        self.binarize_button.setEnabled(False)
        action_layout.addWidget(self.binarize_button)
        # Save button
        self.save_button = QToolButton()
        self.save_button.setIcon(icons.get("save"))
        self.save_button.setToolTip("Save the current result")
        self.save_button.setFixedSize(30, 30)  # A bit wider to accommodate the arrow
        self.save_button.setEnabled(False)
        self.save_button.setPopupMode(QToolButton.InstantPopup)

        save_menu = QMenu(self.save_button)

        self.save_png_action = QAction("Save as PNG Image...", self)
        save_menu.addAction(self.save_png_action)
        self.save_vti_action = QAction("Save as VTI (for ParaView)...", self)
        save_menu.addAction(self.save_vti_action)
        self.save_stl_action = QAction("Save as STL (for CAD)...", self)
        save_menu.addAction(self.save_stl_action)
        self.save_3mf_action = QAction("Save as 3MF (for 3D printing)...", self)
        save_menu.addAction(self.save_3mf_action)

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
