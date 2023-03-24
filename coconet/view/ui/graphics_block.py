"""
Module graphics_block.py

This module contains the graphics elements of Block objects for representing the layers,
the IO and the properties

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtWidgets import QGraphicsItem, QWidget


class GraphicsBlock(QGraphicsItem):
    def __init__(self, block: 'Block'):
        super().__init__()
        # Reference to the block
        self.block_ref = block

        # Content widget
        self.content = None

    def set_content(self, wdg: 'BlockContentWidget'):
        self.content = wdg


class BlockContentWidget(QWidget):
    def __init__(self, block: 'Block', build_dict: dict, parent=None):
        super().__init__(parent)
