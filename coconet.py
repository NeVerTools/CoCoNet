import ctypes
import platform
import sys

from PyQt6.QtWidgets import QApplication

from coconet.main_window import CoCoNetWindow
from scripts import cli

if __name__ == "__main__":

    # GUI mode

    if len(sys.argv) == 1:
        APP_ID = u'org.neuralverification.coconet.2.0'

        # Set taskbar icon on Windows
        if platform.system() == 'Windows':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
            sys.argv += ['-platform', 'windows:darkmode=2']  # TODO remove with styling

        app = QApplication(sys.argv)
        window = CoCoNetWindow()
        window.show()
        sys.exit(app.exec())

    # CLI mode

    elif len(sys.argv) == 2 and sys.argv[1] == "-h":
        cli.show_help()
    elif len(sys.argv) == 3 and sys.argv[1] == "-check":
        cli.check_vnnlib_compliance(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "-convert":
        cli.convert_to_onnx(sys.argv[2])
    else:
        cli.show_help()
