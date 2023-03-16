"""
Module main_window.py

This module contains the QMainWindow class CoCoNetWindow.

Author: Andrea Gimelli, Giacomo Rosato

"""
from PyQt6.QtGui import QIcon, QGuiApplication
from PyQt6.QtWidgets import QMainWindow

from coconet import APP_NAME, ROOT_DIR


class CoCoNetWindow(QMainWindow):
    """
    This class initializes the name of the application, the menu bar and
    a unique CoCoNetWidget object. The menu bar is made of four submenus
    ('File', 'Edit', 'View', 'Help') and each submenu has its own set of
    actions.

    Attributes
    ----------


    Methods
    ----------


    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(ROOT_DIR + '/resources/icons/logo_square.png'))

        # Frame window to center
        self.setGeometry(0, 0, 1024, 736)
        shadow = self.frameGeometry()
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        shadow.moveCenter(center)
        self.move(shadow.topLeft())

        # Create the menu_bar
        self.menu_bar = self.menuBar()
