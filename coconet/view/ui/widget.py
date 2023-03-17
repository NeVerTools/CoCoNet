"""
Module widget.py

This module contains the QWidget class CoCoNetWidget.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout


class CoCoNetWidget(QWidget):
    """
    This class initializes the main layout of the application containing the toolbar
    on the left and the scene on the right.

    """

    def __init__(self, main_window: 'CoCoNetWindow', parent=None):
        super().__init__(parent)

        # Reference to the main window
        self.main_wnd_ref = main_window

        # Widget layout
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
