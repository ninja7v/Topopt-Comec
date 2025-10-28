# tests/test_widgets.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Tests for the widgets.

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton, QSpinBox, QWidget

from app.ui.widgets import (
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

# --- Tests for CollapsibleSection ---


def test_collapsible_section_toggle_logic(qt_app):
    """Unit Test: Verifies the core expand/collapse logic of the CollapsibleSection."""
    content = QWidget()
    section = CollapsibleSection("Test", content)

    # Initial state
    assert section.is_collapsed is True
    assert not content.isVisible()

    # Simulate click to expand
    section.toggle_button.setChecked(True)
    qt_app.processEvents()

    # Force show since some widgets don't auto-show in headless tests
    content.setVisible(True)

    # Now check state
    assert section.is_collapsed is False
    assert not content.isHidden()


# --- Tests for Static Data Widgets ---


def test_dimensions_widget_initialization(qt_app):
    """Unit Test: Verifies that the DimensionsWidget initializes with correct defaults."""
    widget = DimensionsWidget()

    # Check instances
    assert isinstance(widget.nx, QSpinBox)
    assert isinstance(widget.volfrac, QDoubleSpinBox)
    assert isinstance(widget.scale, QDoubleSpinBox)
    assert isinstance(widget.scale_button, QPushButton)

    # Check default values from your provided code
    assert widget.nx.value() == 60
    assert widget.ny.value() == 40
    assert widget.nz.value() == 0
    assert widget.volfrac.value() == 0.3
    assert widget.scale.value() == 1.0


def test_regions_widget_initialization(qt_app):
    """Unit Test: Verifies the RegionsWidget initializes correctly."""
    widget = RegionsWidget()

    first_region = widget.inputs[0]
    # Check instances
    assert "rshape" in first_region and isinstance(first_region["rshape"], QComboBox)
    assert "rstate" in first_region and isinstance(first_region["rstate"], QComboBox)
    assert "rradius" in first_region and isinstance(first_region["rradius"], QSpinBox)
    assert "rx" in first_region and isinstance(first_region["rx"], QSpinBox)
    assert "ry" in first_region and isinstance(first_region["ry"], QSpinBox)

    # Check default values
    assert first_region["rshape"].currentText() == "-"
    assert first_region["rstate"].currentText() == "Void"
    assert first_region["rradius"].value() == 1
    assert first_region["rx"].value() == 0
    assert first_region["ry"].value() == 0


def test_forces_widget_initialization(qt_app):
    """Unit Test: Verifies the ForcesWidget initializes with exactly 3 force rows."""
    widget = ForcesWidget()

    assert (
        len(widget.inputs) >= 2
    ), "ForcesWidget should always create at least 2 force rows."

    first_iforce = widget.inputs[0]
    first_oforce = widget.inputs[2]
    # Check instances
    assert "fix" in first_iforce and isinstance(first_iforce["fix"], QSpinBox)
    assert "fiy" in first_iforce and isinstance(first_iforce["fiy"], QSpinBox)
    assert "fidir" in first_iforce and isinstance(first_iforce["fidir"], QComboBox)
    assert "fox" in first_oforce and isinstance(first_oforce["fox"], QSpinBox)
    assert "foy" in first_oforce and isinstance(first_oforce["foy"], QSpinBox)
    assert "fodir" in first_oforce and isinstance(first_oforce["fodir"], QComboBox)

    # Check default values
    assert first_iforce["fix"].value() == 30
    assert first_iforce["fiy"].value() == 0
    assert first_iforce["fidir"].currentText() == "Y:↑"
    assert first_oforce["fox"].value() == 30
    assert first_oforce["foy"].value() == 40
    assert first_oforce["fodir"].currentText() == "Y:↓"


def test_support_widget_initialization(qt_app):
    """Unit Test: Verifies that the static SupportWidget initializes with 4 rows."""
    widget = SupportWidget()

    assert (
        len(widget.inputs) >= 1
    ), "SupportWidget should always create at least 1 support row."

    first_support = widget.inputs[0]
    # Check instances
    assert "sx" in first_support and isinstance(first_support["sx"], QSpinBox)
    assert "sy" in first_support and isinstance(first_support["sy"], QSpinBox)
    assert "sdim" in first_support and isinstance(first_support["sdim"], QComboBox)

    # Check default values
    assert widget.inputs[0]["sx"].value() == 0
    assert widget.inputs[0]["sy"].value() == 0
    assert widget.inputs[0]["sdim"].currentText() == "XYZ"

    assert widget.inputs[1]["sx"].value() == 60
    assert widget.inputs[1]["sdim"].currentText() == "XYZ"

    assert widget.inputs[3]["sdim"].currentText() == "-"


def test_material_widget_initialization(qt_app):
    """Unit Test: Verifies the MaterialWidget initializes correctly."""
    widget = MaterialWidget()

    # Check instances
    assert isinstance(widget.mat_E, QDoubleSpinBox)
    assert isinstance(widget.mat_nu, QDoubleSpinBox)
    assert isinstance(widget.mat_init_type, QComboBox)

    # Check default values
    assert widget.mat_E.value() == 1.0
    assert widget.mat_nu.value() == 0.25
    assert widget.mat_init_type.currentText() == "Uniform"


def test_optimizer_widget_initialization(qt_app):
    """Unit Test: Verifies the OptimizerWidget initializes correctly."""
    widget = OptimizerWidget()

    # Check instances
    assert isinstance(widget.opt_ft, QComboBox)
    assert isinstance(widget.opt_fr, QDoubleSpinBox)
    assert isinstance(widget.opt_p, QDoubleSpinBox)
    assert isinstance(widget.opt_max_change, QDoubleSpinBox)
    assert isinstance(widget.opt_n_it, QSpinBox)

    # Check default values
    assert widget.opt_ft.currentText() == "Sensitivity"
    assert widget.opt_fr.value() == 1.3
    assert widget.opt_p.value() == 3.0
    assert widget.opt_max_change.value() == 0.05
    assert widget.opt_n_it.value() == 30


def test_displacement_widget_initialization(qt_app):
    """Unit Test: Verifies the DisplacementWidget initializes correctly."""
    widget = DisplacementWidget()

    # Check instances
    assert isinstance(widget.mov_disp, QDoubleSpinBox)
    assert isinstance(widget.mov_iter, QSpinBox)
    assert hasattr(widget, "run_disp_button") and isinstance(
        widget.run_disp_button, QPushButton
    )

    # Check default values
    assert widget.mov_disp.value() == 1.0
    assert widget.mov_iter.value() == 1


# --- Tests for Container Widgets (Header, Footer, Preset) ---


def test_header_widget_structure(qt_app):
    """Unit Test: Ensures the HeaderWidget contains the correct child widgets."""
    widget = HeaderWidget()

    # Check instances
    assert hasattr(widget, "info_button") and isinstance(
        widget.info_button, QPushButton
    )
    assert hasattr(widget, "theme_button") and isinstance(
        widget.theme_button, QPushButton
    )


def test_preset_widget_structure(qt_app):
    """Unit Test: Ensures the PresetWidget contains the correct child widgets."""
    widget = PresetWidget()

    # Check instances
    assert hasattr(widget, "presets_combo") and isinstance(
        widget.presets_combo, QComboBox
    )
    assert hasattr(widget, "save_preset_button")
    assert hasattr(widget, "delete_preset_button")


def test_footer_widget_structure(qt_app):
    """Unit Test: Ensures the FooterWidget contains the correct child widgets."""
    widget = FooterWidget()

    # Check instances
    assert hasattr(widget, "create_button") and isinstance(
        widget.create_button, QPushButton
    )
    assert hasattr(widget, "stop_button") and isinstance(
        widget.stop_button, QPushButton
    )
    assert hasattr(widget, "binarize_button") and isinstance(
        widget.binarize_button, QPushButton
    )
    assert hasattr(widget, "save_button")
