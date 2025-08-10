# tests/conftest.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Fixture for the tests.

import pytest
import matplotlib

# Use a non-interactive backend for tests to prevent warnings and errors
matplotlib.use('Agg')

# You can also move your qt_app fixture here to make it available to all test files
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qt_app():
    """Fixture to create a QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app