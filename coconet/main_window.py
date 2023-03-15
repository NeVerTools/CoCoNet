"""
Module main_window.py

This module contains the QMainWindow class CoCoNetWindow.

Author: Andrea Gimelli, Giacomo Rosato

"""
from PyQt6.QtWidgets import QMainWindow


class CoCoNetWindow(QMainWindow):
    """
    This class initializes the name of the application, the menu bar and
    a unique CoCoNetWidget object. The menu bar is made of three submenus
    ('File', 'Edit', 'View') and each submenu has its own set of actions.

    Attributes
    ----------


    Methods
    ----------


    """

    def __init__(self):
        super().__init__()
