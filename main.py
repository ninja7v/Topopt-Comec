# main.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Entry point of TopoptComec.

import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle
from PySide6.QtSvg import (  # noqa: F401
    QSvgRenderer,
)  # make sure SVG support is available
from app.ui.resource_path_finder import resource_path


def main():
    """Initializes and runs the Qt application."""
    if len(sys.argv) > 1:
        # CLI mode
        from app.cli import run_cli

        run_cli()
    else:
        # GUI mode
        app = QApplication(sys.argv)

        icon_path = resource_path("icons") / "window_icon.svg"
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
        else:
            print(
                f"Warning: Window icon not found at {icon_path}. Using a built-in icon."
            )
            fallback_icon = app.style().standardIcon(
                QStyle.StandardPixmap.SP_ComputerIcon
            )
            app.setWindowIcon(fallback_icon)

        # Now that the app exists, import the main window
        from app.ui.main_window import MainWindow

        window = MainWindow()
        window.show()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
