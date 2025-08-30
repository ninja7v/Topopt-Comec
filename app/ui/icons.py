# app/ui/icons.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Centralized icon management for the application.

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle, QApplication
from pathlib import Path

class IconProvider:
    """Provides standard Qt icons for the UI."""
    def __init__(self):
        self.style = QApplication.style()
        self.theme = 'light'

    def set_theme(self, theme_name: str):
        """Sets the current theme ('light' or 'dark'). Called by the MainWindow."""
        self.theme = theme_name

    def get(self, icon_name: str) -> QIcon:
        """
        Retrieves a QIcon by its name. Tries to load a custom icon from assets first,
        falls back to standard Qt icons if not found.
        """
        # 1. Try to find a themed icon file
        extensions = ['svg', 'png', 'jpg'] # Try .svg first, then .png, then .jpg
        icon_dir = Path("icons")
        for ext in extensions:
            themed_path = icon_dir / f"{icon_name}_{self.theme}.{ext}"
            if themed_path.is_file():
                return QIcon(str(themed_path))

        # 2. If not found, try to find a generic (non-themed) icon file
        for ext in extensions:
            generic_path = icon_dir / f"{icon_name}.{ext}"
            if generic_path.is_file():
                return QIcon(str(generic_path))

        # 3. If no file is found, fall back to built-in Qt icons
        if self.style is None:
            # Ensure the style is initialized, especially for tests
            self.style = QApplication.instance().style()
        
        icon_map = {
            'save': QStyle.StandardPixmap.SP_DialogSaveButton,
            'delete': QStyle.StandardPixmap.SP_TrashIcon,
            'eye_open': QStyle.StandardPixmap.SP_DialogYesButton,
            'eye_closed': QStyle.StandardPixmap.SP_DialogNoButton,
            'arrow_right': QStyle.StandardPixmap.SP_TitleBarShadeButton,
            'arrow_down': QStyle.StandardPixmap.SP_TitleBarUnshadeButton,
            'create': QStyle.StandardPixmap.SP_MediaPlay,
            'folder': QStyle.StandardPixmap.SP_DirOpenIcon,
            'color': QStyle.StandardPixmap.SP_CustomBase,
            'window': QStyle.StandardPixmap.SP_ComputerIcon,
            'sun': QStyle.StandardPixmap.SP_TitleBarMaxButton,
            'moon': QStyle.StandardPixmap.SP_TitleBarMaxButton,
            'info' : QStyle.StandardPixmap.SP_MessageBoxInformation,
            'binarize': QStyle.StandardPixmap.SP_DialogApplyButton,
            'stop': QStyle.StandardPixmap.SP_MediaStop,
            'move': QStyle.StandardPixmap.SP_ArrowRight,
            'reset': QStyle.StandardPixmap.SP_BrowserReload,
        }
        pixmap = icon_map.get(icon_name)
        if pixmap:
            return self.style.standardIcon(pixmap)

        print(f"Warning: Icon '{icon_name}' not found as custom or built-in.")
        return QIcon()  # Return empty icon if not found

# Global instance for easy access
icons = IconProvider()