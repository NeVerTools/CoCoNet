"""
Module graphics_view.py

This module contains the GraphicsView class for rendering graphics objects in the viewport

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
from PyQt6.QtWidgets import QGraphicsView


class GraphicsView(QGraphicsView):
    def __init__(self, gr_scene: 'GraphicsScene', parent=None):
        super().__init__(parent)

        # Reference to the graphics scene
        self.gr_scene_ref = gr_scene
