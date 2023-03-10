from scripts import cli
import sys
# from coconet import mainwindow
from PyQt6 import QtWidgets
import ctypes
import os

if __name__ == "__main__":

    if len(sys.argv) == 1:  # GUI
        myappid = u'org.neuralverification.coconet.2.0'
        if os.name == 'nt':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        # app = QtWidgets.QApplication(sys.argv)
        # window = mainwindow.MainWindow()

        # window.show()
        # sys.exit(app.exec_())

    elif len(sys.argv) == 2 and sys.argv[1] == "-h":
        cli.show_help()
    elif len(sys.argv) == 3 and sys.argv[1] == "-check":
        cli.check_vnnlib_compliance(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "-convert":
        cli.convert_to_onnx(sys.argv[2])
    else:
        cli.show_help()
