"""
Module graphics_socket.py

This module contains the graphical representation of a Socket element

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtWidgets import QGraphicsItem


class GraphicsSocket(QGraphicsItem):
    """
    This class provides the graphics representation of a Socket object

    """

    def __init__(self, socket: 'Socket', parent=None):
        super().__init__(parent)

        self.socket_ref = socket
