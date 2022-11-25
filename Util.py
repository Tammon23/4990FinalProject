from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QFontMetrics, QIcon
from PyQt6.QtWidgets import QMessageBox, QLabel


class MyLabel(QLabel):
    def paintEvent(self, event):
        painter = QPainter(self)

        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())

        painter.drawText(self.rect(), self.alignment(), elided)


class NNException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Resources(Enum):
    MOON_ICON = "Resources/moon.png"
    SUN_ICON = "Resources/sun.png"
    LOGO = "Resources/logo.png"


class ErrorMessageBox:
    def __init__(self, text, title="Error", info="", warnIcon=False):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical if warnIcon else QMessageBox.Icon.Warning)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setWindowTitle(title)
        msg.setWindowIcon(QIcon(Resources.LOGO.value))
        msg.exec()
