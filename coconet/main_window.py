"""
Module main_window.py

This module contains the QMainWindow class CoCoNetWindow.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtGui import QIcon, QGuiApplication
from PyQt6.QtWidgets import QMainWindow

from coconet import APP_NAME, ROOT_DIR
from coconet.view.ui.widget import CoCoNetWidget


class CoCoNetWindow(QMainWindow):
    """
    This class initializes the name of the application, the menu bar and
    a unique CoCoNetWidget object. The menu bar is made of four submenus
    ('File', 'Edit', 'View', 'Help') and each submenu has its own set of
    actions.

    Attributes
    ----------
    menu_bar : QMenuBar
        Qt menubar initializer
    editor_widget : CoCoNetWidget
        Main widget of the application displayed in the window

    Methods
    ----------
    create_menu()
        This method creates the QActions and the submenus
    change_window_title()
        This method changes the title of the window

    """

    def __init__(self):
        super().__init__()

        # Create the main widget
        self.editor_widget = CoCoNetWidget(self)
        self.setCentralWidget(self.editor_widget)

        # Create the menu_bar
        self.menu_bar = self.menuBar()
        self.create_menu()

        # Init window and toolbars
        self.init_ui()
        self.init_toolbars()

    def create_menu(self):
        pass

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(ROOT_DIR + '/resources/icons/logo_square.png'))

        # Frame window to center
        self.setGeometry(0, 0, 1024, 736)
        frame = self.frameGeometry()
        frame.moveCenter(QGuiApplication.primaryScreen().availableGeometry().center())
        self.move(frame.topLeft())

    def init_toolbars(self):
        pass
