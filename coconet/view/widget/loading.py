from random import randint

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QProgressBar


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
