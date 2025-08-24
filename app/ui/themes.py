# app/ui/themes.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Contains stylesheets for the Light and Dark themes.

# Light theme
LIGHT_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #F0F0F0;
        color: #000000;
        font-family: Arial;
    }
    QMainWindow {
        background-color: #F0F0F0;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        padding: 1px;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: #E1E1E1;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        padding: 3px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #E8E8E8;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #D0D0D0;
    }
    
    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #E0E0E0;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
    }
    #collapsibleTitleLabel { font-weight: bold; }
    #collapsibleContent { 
        border: 1px solid #E0E0E0;
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1px;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #C0C0C0;
    }
    QProgressBar {
        border: 1px solid grey;
        border-radius: 5px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #0078D7;
        width: 10px;
    }
    QFrame#presetFrame {
        border: 1px solid #D0D0D0; /* A light gray border */
        border-radius: 4px;
    }
"""

# Dark theme
DARK_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #2E2E2E;
        color: #E0E0E0;
        font-family: Arial;
    }
    QMainWindow {
        background-color: #2E2E2E;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #3C3C3C;
        border: 1px solid #5A5A5A;
        border-radius: 4px;
        padding: 1px;
        color: #E0E0E0;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: #4A4A4A;
        border: 1px solid #5A5A5A;
        border-radius: 4px;
        padding: 3px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #555555;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #3A3A3A;
    }

    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #3C3C3C;
        border: 1px solid #555555;
        border-radius: 4px;
    }
    #collapsibleTitleLabel { font-weight: bold; }
    #collapsibleContent { 
        border: 1px solid #5A5A5A;
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1px;
        background-color: #333333;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #5A5A5A;
    }
    QProgressBar {
        border: 1px solid #5A5A5A;
        border-radius: 5px;
        text-align: center;
        color: #E0E0E0;
    }
    QProgressBar::chunk {
        background-color: #0078D7;
        width: 10px;
    }
    QFrame#presetFrame {
        border: 1px solid #555555; /* A dark gray border, matching other elements */
        border-radius: 4px;
    }
"""