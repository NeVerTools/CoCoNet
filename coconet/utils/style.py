"""
Module style.py

This module contains utility methods for the styling of graphics objects

Author: Andrea Gimelli, Giacomo Rosato

"""

from PyQt6.QtCore import QFile
from PyQt6.QtWidgets import QApplication


def loadStylesheet(filename: str):
    """
    Loads a qss stylesheet to the current QApplication instance

    Parameters
    ----------
    filename : str
        Path to the qss file

    """

    file = QFile(filename)
    file.open(QFile.ReadOnly | QFile.Text)
    stylesheet = file.readAll()
    QApplication.instance().setStyleSheet(str(stylesheet, encoding='utf-8'))


def loadStylesheets(*args):
    """
    Loads multiple qss stylesheets. Concatenates them together and applies
    the final stylesheet to the current QApplication instance

    Parameters
    ----------
    *args : *str
        Multiple paths to qss files

    """

    res = ''
    for arg in args:
        file = QFile(arg)
        file.open(QFile.ReadOnly | QFile.Text)
        stylesheet = file.readAll()
        res += "\n" + str(stylesheet, encoding='utf-8')
    QApplication.instance().setStyleSheet(res)
