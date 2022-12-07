from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QLineEdit, QPlainTextEdit, QPushButton

import coconet.view.styles as style


class CustomLabel(QLabel):
    def __init__(self, text: str = '', color: str = style.WHITE, primary: bool = False, alignment=Qt.AlignLeft):
        super(CustomLabel, self).__init__(text)
        if primary:
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet(style.NODE_LABEL_STYLE)
        else:
            self.setAlignment(alignment)
            self.setStyleSheet('color: ' + color + ';' +
                               'border: none;' +
                               'padding: 2px 0px 2px 2px;')


class CustomComboBox(QComboBox):
    def __init__(self, color: str = style.WHITE):
        super(CustomComboBox, self).__init__()
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: ' + style.GREY_2 + ';' +
                           'border: none;' +
                           'padding: 2px;')


class CustomTextBox(QLineEdit):
    def __init__(self, text: str = '', color: str = style.WHITE):
        super(CustomTextBox, self).__init__()
        self.setText(text)
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: ' + style.GREY_2 + ';' +
                           'border: none;' +
                           'padding: 2px;')


class CustomTextArea(QPlainTextEdit):
    def __init__(self, color: str = style.WHITE, parent=None):
        super(CustomTextArea, self).__init__(parent)
        self.setStyleSheet('color: ' + color + ';' +
                           'background-color: ' + style.GREY_2 + ';' +
                           'border: none;' +
                           'padding: 2px;' +
                           'QPlainTextEdit::placeholder {' +
                           'color: ' + style.GREY_4 + ';' +
                           '}')


class CustomButton(QPushButton):
    def __init__(self, text: str = '', primary: bool = False):
        super(CustomButton, self).__init__(text)
        if primary:
            self.setStyleSheet(style.PRIMARY_BUTTON_STYLE)
            self.setDefault(True)
        else:
            self.setStyleSheet(style.BUTTON_STYLE)
            self.setDefault(False)
