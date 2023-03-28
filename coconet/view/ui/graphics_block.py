"""
Module graphics_block.py

This module contains the graphics elements of Block objects for representing the layers,
the IO and the properties

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor, QFont, QBrush, QPainterPath
from PyQt6.QtWidgets import QGraphicsItem, QWidget, QGraphicsProxyWidget, QGraphicsTextItem

import coconet.resources.styling.dimension as dim
import coconet.resources.styling.palette as palette
from coconet import get_classname


class GraphicsBlock(QGraphicsItem):
    def __init__(self, block: 'Block'):
        super().__init__()
        # Reference to the block
        self.block_ref = block

        # Hover flag
        self.hover = False

        # Content widget
        self.content = None
        self.width = dim.BLOCK_BASE_WIDTH
        self.height = dim.BLOCK_BASE_HEIGHT

        # Init graphics content
        self.title_item = QGraphicsTextItem(self)
        self.graphics_content = QGraphicsProxyWidget(self)
        self.init_title()
        self.init_graphics_content()

        # Style parameters
        self.color_scheme = []
        self.init_colors()
        self._pen_default = QPen(QColor(self.color_scheme[0]))
        self._pen_default.setWidth(2)
        self._pen_hovered = QPen(QColor(self.color_scheme[1]))
        self._pen_hovered.setWidth(2)
        self._pen_selected = QPen(QColor(self.color_scheme[2]))
        self._pen_selected.setWidth(3)
        self._pen_selected.setStyle(Qt.PenStyle.DotLine)
        self._brush_title = QBrush(QColor(self.color_scheme[3]))
        self._brush_background = QBrush(QColor(self.color_scheme[4]))

        self.init_flags()

    def init_title(self):
        """
        This method sets up the title widget

        """

        self.title_item.setDefaultTextColor(QColor(palette.WHITE))
        self.title_item.setFont(QFont('Arial', 10))
        self.title_item.setPos(dim.TITLE_PAD, 0)
        self.title_item.setPlainText(self.block_ref.title)
        self.title_item.setTextWidth(self.width - 2 * dim.TITLE_PAD)

    def init_colors(self):
        """
        This method sets up the color scheme of the block
        depending on the block type

        """

        if get_classname(self.block_ref) == 'LayerBlock':
            self.color_scheme = [palette.DARK_BLUE, palette.BLUE, palette.BLUE,
                                 palette.DARK_BLUE, palette.DARK_GREY]
        elif get_classname(self.block_ref) == 'PropertyBlock':
            self.color_scheme = [palette.DARK_ORANGE, palette.ORANGE, palette.ORANGE,
                                 palette.DARK_ORANGE, palette.DARK_GREY]
        elif get_classname(self.block_ref) == 'FunctionalBlock':
            self.color_scheme = [palette.GREY, palette.LIGHT_GREY, palette.LIGHT_GREY,
                                 palette.GREY, palette.DARK_GREY]

    def init_graphics_content(self):
        """
        This method sets up the graphics properties of the block
        depending on the content

        """

        if self.block_ref.has_parameters():
            self.width = dim.BLOCK_PARAM_WIDTH
        elif get_classname(self.block_ref) == 'PropertyBlock':
            self.width = dim.BLOCK_PROPERTY_WIDTH

        if self.content is not None:
            self.content.setGeometry(dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT + dim.EDGE_ROUNDNESS,
                                     self.width - 2 * dim.EDGE_ROUNDNESS, 0)
            self.graphics_content.setWidget(self.content)

            self.width = self.content.size().width() + 2 * dim.EDGE_ROUNDNESS
            self.height = self.content.size.height() + 2 * dim.EDGE_ROUNDNESS + dim.TITLE_HEIGHT

    def init_flags(self):
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

    def open_dock_params(self):
        self.block_ref.scene_ref.editor_widget_ref.main_wnd_ref.load_inspector(self.block_ref)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.hover = False
        self.update()

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        super().mouseMoveEvent(event)

        for block in self.scene_ref.blocks.values():
            if block.isSelected():
                block.updateConnectedEdges()

        self.block_ref.updateConnectedEdges()

    def mouseDoubleClickEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.open_dock_params()

    def paint(self, painter: 'QPainter', option: 'QStyleOptionGraphicsItem', widget=None) -> None:
        """
        This method draws the graphicsBlock item. It is a rounded rectangle divided in 3 sections:

        Outline section: draw the contours of the block
        Title section: a darker rectangle in which lays the title
        Content section: container for the block parameters

        """

        # Title section
        path_title = QPainterPath()
        path_title.setFillRule(Qt.FillRule.WindingFill)
        path_title.addRoundedRect(0, 0, self.width, dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)

        # Remove bottom rounded corners for title box
        # (add the following line if you finally want to remove content on block with no parameters)
        # if self.hasParameters != 0:
        path_title.addRect(0, dim.TITLE_HEIGHT - dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        path_title.addRect(self.width - dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT - dim.EDGE_ROUNDNESS,
                           dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_title)
        painter.drawPath(path_title.simplified())

        # Content
        path_content = QPainterPath()
        path_content.setFillRule(Qt.FillRule.WindingFill)
        path_content.addRoundedRect(0, dim.TITLE_HEIGHT, self.width,
                                    self.height - dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)

        # Remove bottom rounded corners for content box
        # left
        path_content.addRect(0, dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS / 2)
        # right
        path_content.addRect(self.width - dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT,
                             dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS / 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_background)
        painter.drawPath(path_content.simplified())

        # Outline
        path_outline = QPainterPath()
        path_outline.addRoundedRect(0, 0, self.width, self.height, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        painter.setPen(self._pen_default if not self.isSelected() else self._pen_selected)

        if self.hover and not self.isSelected():
            painter.setPen(self._pen_hovered)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_outline.simplified())

    def boundingRect(self) -> QtCore.QRectF:
        """
        Defines the Qt bounding rectangle

        Returns
        ----------
        QRectF
            The area in which the click triggers the item

        """

        return QtCore.QRectF(0, 0, self.width, self.height).normalized()


class BlockContentWidget(QWidget):
    def __init__(self, block: 'Block', build_dict: dict, parent=None):
        super().__init__(parent)
