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
        self.presets_combo = QComboBox()
        self.presets_combo.setToolTip("Load a saved set of parameters")
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
        self.nx = QSpinBox()
        self.nx.setRange(5, 1000)
        self.nx.setValue(60)
        self.nx.setMaximumWidth(70)
        self.nx.setToolTip("X")
        size_layout.addWidget(self.nx)
        size_layout.addWidget(QLabel("x"))
        self.ny = QSpinBox()
        self.ny.setRange(5, 1000)
        self.ny.setValue(40)
        self.ny.setMaximumWidth(70)
        self.ny.setToolTip("Y")
        size_layout.addWidget(self.ny)
        size_layout.addWidget(QLabel("x"))
        self.nz = QSpinBox()
        self.nz.setRange(0, 1000)
        self.nz.setValue(0)
        self.nz.setMaximumWidth(70)
        self.nz.setToolTip("Z → set to 0 for a 2D problem")
        size_layout.addWidget(self.nz)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        # Volume Fraction
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Vol. Frac:"))
        self.volfrac = QDoubleSpinBox()
        self.volfrac.setRange(0.05, 0.8)
        self.volfrac.setValue(0.3)
        self.volfrac.setSingleStep(0.05)
        self.volfrac.setToolTip("Volume Fraction")
        vol_layout.addWidget(self.volfrac)
        vol_layout.addStretch()
        layout.addLayout(vol_layout)
        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.5, 5)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.5)
        self.scale.setToolTip("Scale every component by this factor")
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
        rshape = QComboBox()
        rshape.addItems(["-", "□", "◯"])
        rshape.setCurrentIndex(0)
        rshape.setToolTip("Shape of the region")
        line1_layout.addWidget(rshape)
        rstate = QComboBox()
        rstate.addItems(["Void", "Filled"])
        rstate.setCurrentIndex(0)
        rstate.setToolTip("State of the region")
        line1_layout.addWidget(rstate)
        # Radius
        line1_layout.addWidget(QLabel("Radius:"))
        rradius = QSpinBox()
        rradius.setRange(1, 100)
        rradius.setValue(3)
        rradius.setMaximumWidth(60)
        rradius.setToolTip("Radius")
        line1_layout.addWidget(rradius)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # --- Line 2: "Center" [X] [Y] [Z] ---
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)  # Align with line above
        line2_layout.addWidget(QLabel("Center:"))
        rx = QSpinBox()
        rx.setRange(0, 1000)
        rx.setValue(pos[0])
        rx.setMaximumWidth(60)
        rx.setToolTip("X")
        line2_layout.addWidget(rx)
        ry = QSpinBox()
        ry.setRange(0, 1000)
        ry.setValue(pos[1])
        ry.setMaximumWidth(60)
        ry.setToolTip("Y")
        line2_layout.addWidget(ry)
        rz = QSpinBox()
        rz.setRange(0, 1000)
        rz.setValue(pos[2])
        rz.setMaximumWidth(60)
        rz.setToolTip("Z")
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
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        arrows = ["-", "X:→", "X:←", "Y:↑", "Y:↓", "Z:<", "Z:>"]
        nb_default_input_forces = 2
        iorigins = [[30, 0, 0], [0, 0, 0]]
        iarrows = [3, 0]
        inorms = [0.01, 0.0]
        nb_default_output_forces = 2
        oorigins = [[30, 40, 0], [0, 0, 0]]
        oarrows = [4, 0]
        onorms = [0.01, 0.0]

        force_input_label = QLabel("Input")
        force_input_label.setStyleSheet("color: red;")
        main_layout.addWidget(force_input_label)
        for i in range(nb_default_input_forces):
            grid = QGridLayout()
            grid.setColumnStretch(1, 1)  # Allow input column to expand
            # Force position Row
            origin_label = QLabel("Origin:")
            grid.addWidget(origin_label, 0, 0)
            pos_layout = QHBoxLayout()
            fx = QSpinBox()
            fx.setRange(0, 1000)
            fx.setValue(iorigins[i][0])
            fx.setMaximumWidth(70)
            fx.setToolTip("X")
            pos_layout.addWidget(fx)
            fy = QSpinBox()
            fy.setRange(0, 1000)
            fy.setValue(iorigins[i][1])
            fy.setMaximumWidth(70)
            fy.setToolTip("Y")
            pos_layout.addWidget(fy)
            fz = QSpinBox()
            fz.setRange(0, 1000)
            fz.setValue(iorigins[i][2])
            fz.setMaximumWidth(70)
            fz.setToolTip("Z")
            pos_layout.addWidget(fz)
            grid.addLayout(pos_layout, 0, 1)
            # Direction
            dir_layout = QHBoxLayout()
            dir_layout.addWidget(QLabel("Dir:"))
            fdir = QComboBox()
            fdir.addItems(arrows)
            fdir.setCurrentIndex(iarrows[i])
            fdir.setToolTip("Force direction")
            dir_layout.addWidget(fdir)
            dir_layout.addSpacing(20)
            # Force spring
            dir_layout.addWidget(QLabel("Spring (N/m):"))
            fnorm = QDoubleSpinBox()
            fnorm.setRange(0, 10)
            fnorm.setSingleStep(0.01)
            fnorm.setValue(inorms[i])
            fnorm.setToolTip("Force magnitude for input, spring stiffness for output")
            dir_layout.addWidget(fnorm)
            dir_layout.addStretch()
            grid.addLayout(dir_layout, 1, 0, 1, 2)  # Span across both columns

            # Add this force's grid to the main vertical layout
            main_layout.addLayout(grid)

            # Store the widgets in the public 'inputs' list
            self.inputs.append(
                {"fix": fx, "fiy": fy, "fiz": fz, "fidir": fdir, "finorm": fnorm}
            )

            # Add a separator line between force sections
            if i < nb_default_input_forces - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                main_layout.addWidget(line)

        force_output_label = QLabel("Output")
        force_output_label.setStyleSheet("color: blue;")
        main_layout.addWidget(force_output_label)
        for i in range(nb_default_output_forces):
            grid = QGridLayout()
            grid.setColumnStretch(1, 1)  # Allow input column to expand
            # Force position Row
            origin_label = QLabel("Origin:")
            grid.addWidget(origin_label, 0, 0)
            pos_layout = QHBoxLayout()
            fx = QSpinBox()
            fx.setRange(0, 1000)
            fx.setValue(oorigins[i][0])
            fx.setMaximumWidth(70)
            fx.setToolTip("X")
            pos_layout.addWidget(fx)
            fy = QSpinBox()
            fy.setRange(0, 1000)
            fy.setValue(oorigins[i][1])
            fy.setMaximumWidth(70)
            fy.setToolTip("Y")
            pos_layout.addWidget(fy)
            fz = QSpinBox()
            fz.setRange(0, 1000)
            fz.setValue(oorigins[i][2])
            fz.setMaximumWidth(70)
            fz.setToolTip("Z")
            pos_layout.addWidget(fz)
            grid.addLayout(pos_layout, 0, 1)
            # Direction
            dir_layout = QHBoxLayout()
            dir_layout.addWidget(QLabel("Dir:"))
            fdir = QComboBox()
            fdir.addItems(arrows)
            fdir.setCurrentIndex(oarrows[i])
            fdir.setToolTip("Force direction")
            dir_layout.addWidget(fdir)
            dir_layout.addSpacing(20)
            # Force spring
            dir_layout.addWidget(QLabel("Spring (N/m):"))
            fnorm = QDoubleSpinBox()
            fnorm.setRange(0, 10)
            fnorm.setSingleStep(0.01)
            fnorm.setValue(onorms[i])
            fnorm.setToolTip("Force magnitude for input, spring stiffness for output")
            dir_layout.addWidget(fnorm)
            dir_layout.addStretch()
            grid.addLayout(dir_layout, 1, 0, 1, 2)  # Span across both columns

            # Add this force's grid to the main vertical layout
            main_layout.addLayout(grid)

            # Store the widgets in the public 'inputs' list
            self.inputs.append(
                {"fox": fx, "foy": fy, "foz": fz, "fodir": fdir, "fonorm": fnorm}
            )

            # Add a separator line between force sections
            if i < nb_default_output_forces - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                main_layout.addWidget(line)


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
        self.grid = QGridLayout()
        self.grid.setColumnStretch(1, 1)
        self.main_layout.addLayout(self.grid)

        # Add button
        self.add_btn = QPushButton("+ Add Support")
        self.add_btn.clicked.connect(lambda: self.add_support())
        self.add_btn.setToolTip("Add a support")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)

        self.dims = ["-", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]

    def add_support(self, pos=None, dim="XYZ", emit_signal=True):
        row = len(self.inputs)
        if row >= 10:  # safety
            return

        if pos is None:
            pos = [0, 0, 0]

        # Remove button
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this support")
        remove_btn.clicked.connect(lambda checked=False, r=row: self.remove_support(r))
        self.grid.addWidget(remove_btn, row, 0)

        # Position
        pos_layout = QHBoxLayout()
        sx = QSpinBox()
        sx.setRange(0, 1000)
        sx.setValue(pos[0])
        sx.setMaximumWidth(65)
        sx.setToolTip("X")
        sy = QSpinBox()
        sy.setRange(0, 1000)
        sy.setValue(pos[1])
        sy.setMaximumWidth(65)
        sy.setToolTip("Y")
        sz = QSpinBox()
        sz.setRange(0, 1000)
        sz.setValue(pos[2])
        sz.setMaximumWidth(65)
        sz.setToolTip("Z")
        pos_layout.addWidget(sx)
        pos_layout.addWidget(sy)
        pos_layout.addWidget(sz)
        self.grid.addLayout(pos_layout, row, 1)

        # Fixed directions
        sdim = QComboBox()
        sdim.addItems(self.dims)
        sdim.setCurrentText(dim)
        sdim.setToolTip("Fixed direction(s)")
        self.grid.addWidget(sdim, row, 2)

        # Store the widgets in the public 'inputs' list
        data = {"sx": sx, "sy": sy, "sz": sz, "sdim": sdim, "remove_btn": remove_btn}
        self.inputs.append(data)
        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def remove_support(self, row, emit_signal=True):
        if row < 0 or row >= len(self.inputs):
            return

        # Remove widgets
        support = self.inputs.pop(row)
        for w in [
            support["remove_btn"],
            support["sx"],
            support["sy"],
            support["sz"],
            support["sdim"],
        ]:
            w.deleteLater()
            w.setParent(None)

        # Rebuild grid to fix row indices
        self.rebuild_grid()
        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def rebuild_grid(self):
        # Clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                layout = item.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().setParent(None)
                layout.setParent(None)
                layout.deleteLater()

        # Re-add all
        for i, support in enumerate(self.inputs):
            support["remove_btn"].clicked.disconnect()
            support["remove_btn"].clicked.connect(lambda _, r=i: self.remove_support(r))
            self.grid.addWidget(support["remove_btn"], i, 0)
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(support["sx"])
            pos_layout.addWidget(support["sy"])
            pos_layout.addWidget(support["sz"])
            self.grid.addLayout(pos_layout, i, 1)
            self.grid.addWidget(support["sdim"], i, 2)

    def update_remove_buttons(self):
        enabled = len(self.inputs) > 0
        for support in self.inputs:
            support["remove_btn"].setEnabled(enabled)


class MaterialsWidget(QWidget):
    """Custom widget for defining the materials."""

    nbMaterialsChanged = Signal()

    def __init__(self):
        super().__init__()
        self.inputs = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Initialization Type (Global for all materials)
        init_layout = QHBoxLayout()
        init_layout.addWidget(QLabel("Initialization:"))
        self.mat_init_type = QComboBox()
        self.mat_init_type.addItems(["Uniform", "Around activity points", "Random"])
        self.mat_init_type.setCurrentIndex(0)
        self.mat_init_type.setToolTip("Material distribution initialization type")
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
        mat_percent = QSpinBox()
        mat_percent.setRange(0, 100)
        mat_percent.setValue(percent)
        mat_percent.setMaximumWidth(70)
        mat_percent.setSingleStep(5)
        mat_percent.setToolTip("Volume Fraction percentage")
        line1.addWidget(mat_percent)
        line1.addStretch()
        layout.addLayout(line1)

        # Line 2: E [Spin] nu [Spin]
        line2 = QHBoxLayout()
        line2.addSpacing(40)  # Align with line above
        # Young's Modulus
        line2.addWidget(QLabel("E:"))
        mat_E = QDoubleSpinBox()
        mat_E.setRange(0.1, 100.0)
        mat_E.setValue(E)
        mat_E.setMaximumWidth(70)
        mat_E.setToolTip("Young's Modulus, material’s stiffness")
        line2.addWidget(mat_E)
        line2.addSpacing(20)
        # Poisson's Ratio
        line2.addWidget(QLabel("ν:"))
        mat_nu = QDoubleSpinBox()
        mat_nu.setRange(0.0, 0.49)  # 0.5 causes issues in 3D?
        mat_nu.setValue(nu)
        mat_nu.setMaximumWidth(70)
        mat_nu.setSingleStep(0.05)
        mat_nu.setToolTip(
            "Poisson's Ratio, material’s lateral shrinkage relative to its elongation"
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

    def remove_material(self, idx, emit_signal=True):
        if idx < 0 or idx >= len(self.inputs):
            return

        mat = self.inputs.pop(idx)
        mat["container"].deleteLater()
        mat["container"].setParent(None)

        if emit_signal:
            self.nbMaterialsChanged.emit()
        self.update_ui_state()

    def update_ui_state(self):
        # Update Add/Remove buttons state
        can_add = len(self.inputs) < 3
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
        self.opt_ft = QComboBox()
        self.opt_ft.addItems(["Sensitivity", "Density", "None"])
        self.opt_ft.setCurrentIndex(0)
        self.opt_ft.setToolTip(
            "Regularization to avoid checkerboards and mesh dependency\n"
            "Sensitivity: Averages sensitivities to ensure stable and physically meaningful optimization gradients.\n"
            "Density: Smooths the material distribution to avoid checkerboard patterns and mesh dependency."
        )
        layout.addWidget(QLabel("Filter Type:"), 0, 0)
        layout.addWidget(self.opt_ft, 0, 1)
        # Filter Radius
        layout.addWidget(QLabel("Filter Radius:"), 1, 0)
        self.opt_fr = QDoubleSpinBox()
        self.opt_fr.setRange(0.1, 10.0)
        self.opt_fr.setValue(1.3)
        self.opt_fr.setToolTip("Range of the filter coverage")
        layout.addWidget(self.opt_fr, 1, 1)
        # Penalization
        layout.addWidget(QLabel("Penalization:"), 2, 0)
        self.opt_p = QDoubleSpinBox()
        self.opt_p.setRange(1.0, 10.0)
        self.opt_p.setValue(3.0)
        self.opt_p.setToolTip(
            "Exponent in the SIMP method to penalize intermediate densities"
        )
        layout.addWidget(self.opt_p, 2, 1)
        # Eta
        layout.addWidget(QLabel("Eta:"), 3, 0)
        self.opt_eta = QDoubleSpinBox()
        self.opt_eta.setRange(0.05, 1.0)
        self.opt_eta.setValue(0.3)
        self.opt_eta.setSingleStep(0.05)
        self.opt_eta.setToolTip(
            "Damping factor in OC update rule to controls aggressiveness of density updates.\n"
            "Lower eta: slower, more stable convergence; Higher eta: faster, but risk"
        )
        layout.addWidget(self.opt_eta, 3, 1)
        # Max change
        layout.addWidget(QLabel("Max change:"), 4, 0)
        self.opt_max_change = QDoubleSpinBox()
        self.opt_max_change.setRange(0.01, 0.5)
        self.opt_max_change.setValue(0.05)
        self.opt_max_change.setSingleStep(0.05)
        self.opt_max_change.setToolTip(
            "Bound the density change between two iterations to a maximum value"
        )
        layout.addWidget(self.opt_max_change, 4, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 5, 0)
        self.opt_n_it = QSpinBox()
        self.opt_n_it.setRange(1, 100)
        self.opt_n_it.setValue(30)
        self.opt_n_it.setToolTip("Number of optimization iterations")
        layout.addWidget(self.opt_n_it, 5, 1)
        # Solver
        layout.addWidget(QLabel("Solver:"), 6, 0)
        self.opt_solver = QComboBox()
        self.opt_solver.addItems(["Auto", "Direct", "Iterative"])
        self.opt_solver.setCurrentIndex(0)
        self.opt_solver.setToolTip(
            "Solver type for the linear system\n"
            "Auto: Chooses the best solver based on the problem size\n"
            "Direct: Uses LU factorization (spsolve)\n"
            "Iterative: Uses Conjugate Gradient (cg) with preconditioning\n"
        )
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
        self.mov_disp = QDoubleSpinBox()
        self.mov_disp.setRange(0.1, 100.0)
        self.mov_disp.setValue(1.0)
        self.mov_disp.setToolTip("Total scaling factor for the displacement animation")
        layout.addWidget(self.mov_disp, 0, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.mov_iter = QSpinBox()
        self.mov_iter.setRange(1, 20)
        self.mov_iter.setValue(1)
        self.mov_iter.setToolTip(
            "Number of frames in the displacement animation.\n"
            "If more than 1 iteration is set, a more complex function is used since it requires an domain enlargement and interpolations."
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
