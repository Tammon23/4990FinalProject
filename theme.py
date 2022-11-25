from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


class DarkTheme(QPalette):
    """ Some colours chosen from https://stackoverflow.com/a/56851493/19236048"""
    STYLE = "Fusion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
        self.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.PlaceholderText, QColor(102, 102, 102))
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)


class LightTheme(QPalette):
    STYLE = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
