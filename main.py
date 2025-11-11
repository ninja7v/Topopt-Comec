# main.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Entry point of TopOpt-Comec application.

import sys
from pathlib import Path

import darkdetect
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle
from PySide6.QtSvg import (
    QSvgRenderer,
)  # make sure SVG support is available # noqa: F401


def main():
    """Initializes and runs the Qt application."""
    app = QApplication(sys.argv)

    theme = "dark" if darkdetect.isDark() else "light"
    file_name = f"window_icon_{theme}.svg"
    icon_path = Path(__file__).parent / "icons" / file_name
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
    else:
        print(f"Warning: Window icon not found at {icon_path}. Using a built-in icon.")
        fallback_icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        app.setWindowIcon(fallback_icon)

    # Now that the app exists, import the main window
    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
