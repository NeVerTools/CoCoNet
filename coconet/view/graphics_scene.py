"""
Module graphics_scene.py

This module contains the GraphicsScene class for handling graphics objects representation

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtWidgets import QGraphicsScene


class GraphicsScene(QGraphicsScene):
    def __init__(self, scene: 'Scene', parent=None):
        super().__init__(parent)

        # Reference to the scene
        self.scene_ref = scene
