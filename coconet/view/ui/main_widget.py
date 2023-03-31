"""
Module main_widget.py

This module contains the QWidget class CoCoNetWidget.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QSplitter

import coconet.utils.repr as fu
from coconet import RES_DIR
from coconet.model.scene import Scene
from coconet.resources.styling.custom import CustomButton
from coconet.view.components.inspector import InspectorDockToolbar


class CoCoNetWidget(QWidget):
    """
    This class initializes the main layout of the application containing the toolbar
    on the left and the scene on the right.

    """

    def __init__(self, main_window: 'CoCoNetWindow', parent=None):
        super().__init__(parent)

        # Reference to the main window
        self.main_wnd_ref = main_window

        # Style
        with open(RES_DIR + '/styling/qss/style.qss') as qss_file:
            self.setStyleSheet(qss_file.read())

        # Widget layout
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.splitter = QSplitter(self)
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)

        # Objects data from JSON
        self.block_data, self.property_data, self.functional_data = fu.read_json_data()

        # Layers toolbar
        self.layers_toolbar = self.create_layers_toolbar()
        self.inspector = InspectorDockToolbar(self.block_data, self.property_data)

        # Scene
        self.scene = Scene(self)

    def create_layers_toolbar(self) -> QTreeWidget:
        """
        This method creates a QTreeWidget object reading values from block_data

        """

        toolbar_tree = QTreeWidget()
        toolbar_tree.setHeaderHidden(True)

        for i in self.block_data.keys():
            i_item = QTreeWidgetItem([i])
            toolbar_tree.addTopLevelItem(i_item)

            for j in self.block_data[i].keys():
                j_item = QTreeWidgetItem(i_item, [j])
                button = CustomButton(j)
                dict_sign = i + ':' + j
                draw_part = partial(self.add_block_proxy, self.block_data[i][j], dict_sign)
                button.clicked.connect(draw_part)
                toolbar_tree.setItemWidget(j_item, 0, button)

        # Size control
        toolbar_tree.setMinimumWidth(250)
        toolbar_tree.setMaximumWidth(400)
        toolbar_tree.expandAll()

        self.splitter.addWidget(toolbar_tree)
        return toolbar_tree

    def add_block_proxy(self, block_data: dict, block_sign: str):
        """
        Proxy method to add a node in the scene

        """

        self.scene.add_layer_block(block_data, block_sign)

    def save_prompt_dialog(self):
        pass

    def new(self):
        pass

    def open(self):
        pass

    def open_property(self):
        pass

    def save(self, _as: bool = False):
        pass

    def clear(self):
        pass

    def remove_sel(self):
        pass

    def show_inspector(self, block=None):
        self.inspector.display(block)
