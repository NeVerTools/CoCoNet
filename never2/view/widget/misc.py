import logging
from random import randint

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QProgressBar


class LoggerTextBox(logging.Handler):
    """
    This class represents a QPlainTextEdit widget
    used to display logging info. By extending
    logging.Handler it can be used for any logging
    purpose.

    Credits: Todd Vanyo https://stackoverflow.com/users/2623625/todd-vanyo

    Attributes
    ----------
    widget : QPlainTextEdit
        The widget on which the logs are displayed.

    Methods
    ----------
    emit(LogRecord)
        Override of emit() method.

    """

    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setFixedHeight(150)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class ProgressBar(QProgressBar):
    """
    This class is a custom progress bar which shows
    a dynamic loading bar.

    Attributes
    ----------
    timer : QTimer
        Loop iteration duration.

    Methods
    ----------
    onTimeOut()
        Updates the loading bar value.

    """

    def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)

        if self.minimum() != self.maximum():
            self.timer = QTimer(self, timeout=self.onTimeout)
            self.timer.start(randint(1, 3) * 1000)

    def onTimeout(self):
        """
        This method changes the bar value.

        """

        if self.value() >= 100:
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
            return

        self.setValue(self.value() + 1)
