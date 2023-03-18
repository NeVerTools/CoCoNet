"""
Module custom.py

This module defines custom UI components with pre-defined style

Author: Stefano Demarchi

"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QComboBox, QLineEdit, QPlainTextEdit, QPushButton


class CustomButton(QPushButton):
    def __init__(self, text: str = '', primary: bool = False):
        super(CustomButton, self).__init__(text)
        if primary:
            # self.setStyleSheet(style.PRIMARY_BUTTON_STYLE)
            self.setDefault(True)
        else:
            # self.setStyleSheet(style.BUTTON_STYLE)
            self.setDefault(False)


class CustomLabel(QLabel):
    def __init__(self, text: str = '', color: str = 'white', primary: bool = False,
                 alignment=Qt.AlignmentFlag.AlignLeft):
        super(CustomLabel, self).__init__(text)
        # self.setWordWrap(True)
        if primary:
            self.setAlignment(Qt.AlignmentFlag.AlignLeft)
            # self.setStyleSheet(style.NODE_LABEL_STYLE)
        else:
            self.setAlignment(alignment)
            self.setStyleSheet('color: ' + color + ';' +
                               'border: none;' +
                               'padding: 2px 0px 2px 2px;')


class CustomComboBox(QComboBox):
    def __init__(self, color: str = 'white'):
        super(CustomComboBox, self).__init__()
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: grey;' +
                           'border: none;' +
                           'padding: 2px;')


class CustomTextBox(QLineEdit):
    def __init__(self, text: str = '', color: str = 'white'):
        super(CustomTextBox, self).__init__()
        self.setText(text)
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: grey;' +
                           'border: none;' +
                           'padding: 2px;')


class CustomTextArea(QPlainTextEdit):
    def __init__(self, color: str = 'white', parent=None):
        super(CustomTextArea, self).__init__(parent)
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: grey;' +
                           'border: none;' +
                           'padding: 2px;' +
                           'QPlainTextEdit::placeholder {' +
                           'color: grey;' +
                           '}')
