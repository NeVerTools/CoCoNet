"""
Module graphics_scene.py

This module contains the GraphicsScene class for handling graphics objects representation

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
import math

from PyQt6.QtCore import QLine
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsScene

from coconet.resources.styling import palette


class GraphicsScene(QGraphicsScene):
    def __init__(self, scene: 'Scene', parent=None):
        super().__init__(parent)

        # Reference to the scene
        self.scene_ref = scene

        # Create the scene rectangle
        self.scene_width, self.scene_height = 12800, 6400
        self.setSceneRect(-self.scene_width // 2, -self.scene_height // 2, self.scene_width, self.scene_height)

        # Create the background grid
        self.grid_size = 20
        self.grid_square = 5
        self._color_background = QColor(palette.BACKGROUND_GREY)
        self._color_light = QColor(palette.BACKGROUND_LIGHT_LINE_GREY)
        self._color_dark = QColor(palette.BACKGROUND_DARK_LINE_GREY)

        # Pen settings
        self._pen_light = QPen(self._color_light)
        self._pen_light.setWidth(1)
        self._pen_dark = QPen(self._color_dark)
        self._pen_dark.setWidth(2)

        self.setBackgroundBrush(self._color_background)

    def dragMoveEvent(self, event):
        """
        Necessary override for enabling events

        """

        pass

    def drawBackground(self, painter, rect):
        """
        This method draws the background of the scene (setting the color and adding a grid)
        using the painter and a set of QPens

        Parameters
        ----------
        painter : QPainter
            QPainter performs low-level painting on widgets and other paint devices
        rect : QRectF
            A rectangle is normally expressed as a top-left corner and a size

        """

        super().drawBackground(painter, rect)

        # Here we create our grid
        left = int(math.floor(rect.left()))
        right = int(math.ceil(rect.right()))
        top = int(math.floor(rect.top()))
        bottom = int(math.ceil(rect.bottom()))

        first_left = left - (left % self.grid_size)
        first_top = top - (top % self.grid_size)

        # Compute all lines to be drawn
        lines_light, lines_dark = [], []  # position (x1, y1), (x2, y2)

        for x in range(first_left, right, self.grid_size):
            if x % (self.grid_size * self.grid_square) != 0:
                lines_light.append(QLine(x, top, x, bottom))
            else:
                lines_dark.append(QLine(x, top, x, bottom))

        for y in range(first_top, bottom, self.grid_size):
            if y % (self.grid_size * self.grid_square) != 0:
                lines_light.append(QLine(left, y, right, y))
            else:
                lines_dark.append(QLine(left, y, right, y))

        painter.setPen(self._pen_light)
        painter.drawLines(*lines_light)
        painter.setPen(self._pen_dark)
        painter.drawLines(*lines_dark)
