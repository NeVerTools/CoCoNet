"""
Module main_window.py

This module contains the QMainWindow class CoCoNetWindow.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
import json
import webbrowser

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QGuiApplication, QAction, QCloseEvent
from PyQt6.QtWidgets import QMainWindow

from coconet import APP_NAME, RES_DIR
from coconet.view.ui.widget import CoCoNetWidget


def open_guide():
    webbrowser.open('https://neuralverification.org/coconet_guide.html')


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
        self.load_inspector()

    def create_menu(self):
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        menu_actions = dict()  # Dictionary containing the QAction objects linked to the menu items

        with open(RES_DIR + '/json/menu.json') as json_menu:
            menu = json.loads(json_menu.read())

            for menu_item, actions in menu.items():
                entry = self.menu_bar.addMenu(f'&{menu_item}')

                # Read action properties
                for a, v in actions.items():
                    action_item = QAction(a, self)

                    if 'Shortcut' in v.keys():
                        action_item.setShortcut(v['Shortcut'])
                    if v['checkable'] == 'True':
                        action_item.setCheckable(True)
                        action_item.setChecked(True)

                    # Populate dict
                    entry.addAction(action_item)
                    menu_actions[f'{menu_item}:{a}'] = action_item

        if menu_actions:
            # FILE submenu triggers connections
            menu_actions["File:New..."].triggered.connect(self.editor_widget.new)
            menu_actions["File:Open..."].triggered.connect(self.editor_widget.open)
            menu_actions["File:Open property..."].triggered.connect(self.editor_widget.open_property)
            menu_actions["File:Save"].triggered.connect(self.editor_widget.save)
            menu_actions["File:Save as..."].triggered.connect(lambda: self.editor_widget.save(_as=True))
            menu_actions["File:Close"].triggered.connect(self.quit)

            # EDIT submenu triggers connections
            menu_actions["Edit:Clear workspace"].triggered.connect(self.editor_widget.clear)
            menu_actions["Edit:Delete block"].triggered.connect(self.editor_widget.remove_sel)

            # VIEW submenu triggers connections
            # menu_actions["View:Zoom in"].triggered.connect(self.editor_widget.scene.view.zoom_in)
            # menu_actions["View:Zoom out"].triggered.connect(self.editor_widget.scene.view.zoom_out)
            # menu_actions['View:Toggle edges dimensions'].triggered.connect()
            menu_actions["View:Show inspector"].triggered.connect(self.editor_widget.show_inspector)

            # HELP
            menu_actions["Help:Open guide"].triggered.connect(open_guide)

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(RES_DIR + '/icons/logo_square.png'))

        # Frame window to center
        self.setGeometry(0, 0, 1024, 736)
        frame = self.frameGeometry()
        frame.moveCenter(QGuiApplication.primaryScreen().availableGeometry().center())
        self.move(frame.topLeft())

    def load_inspector(self):
        pass

    def quit(self):
        self.editor_widget.save_prompt_dialog()
        self.close()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.editor_widget.save_prompt_dialog()
        a0.accept()
