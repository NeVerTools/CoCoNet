"""
Module graphics_view.py

This module contains the GraphicsView class for rendering graphics objects in the viewport

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QGraphicsView

import coconet.resources.styling.dimension as dim
import coconet.utils.rep as rep
from coconet.model.component.block import LayerBlock, PropertyBlock
from coconet.view.ui.dialog import MessageDialog, MessageType


class GraphicsView(QGraphicsView):
    """
    This class visualizes the contents of the GraphicsScene in a scrollable viewport

    """

    def __init__(self, gr_scene: 'GraphicsScene', parent=None):
        super().__init__(parent)

        # Reference to the graphics scene
        self.gr_scene_ref = gr_scene
        self.setScene(self.gr_scene_ref)
        self.zoom = dim.ZOOM

        self.init_ui()

    def init_ui(self):
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def zoom_in(self):
        self.zoom += dim.ZOOM_STEP
        self.set_scale(dim.ZOOM_IN_FACTOR)

    def zoom_out(self):
        self.zoom -= dim.ZOOM_STEP
        self.set_scale(1 / dim.ZOOM_IN_FACTOR)

    def set_scale(self, factor: float):
        clipped = False
        if self.zoom < dim.ZOOM_RANGE[0]:
            self.zoom = dim.ZOOM_RANGE[0]
            clipped = True
        if self.zoom > dim.ZOOM_RANGE[1]:
            self.zoom = dim.ZOOM_RANGE[1]
            clipped = True

        # Set scene scale
        if not clipped:
            self.scale(factor, factor)

    def check_delete(self):
        """
        This method checks the selected items in the scene and tries to delete them

        """

        delete = True
        sel_ids = []

        for item in self.gr_scene_ref.selectedItems():
            if hasattr(item, 'block_ref'):
                if isinstance(item.block_ref, LayerBlock):
                    sel_ids.append(item.block_ref.id)

        for i in range(len(sel_ids)):
            if self.gr_scene_ref.scene_ref.sequential_list[-2 - i] not in sel_ids:
                delete = False

        if delete:
            self.delete_items(sel_ids)
        else:
            dialog = MessageDialog('It is only allowed to delete blocks at the end of the network',
                                   MessageType.ERROR)
            dialog.exec()

    def delete_items(self, sel_ids: list):
        for block_id in sel_ids:
            block = self.gr_scene_ref.scene_ref.blocks[block_id]
            self.gr_scene_ref.scene_ref.remove_block(block, logic=True)

        for item in self.gr_scene_ref.selectedItems():
            if hasattr(item, 'block_ref'):
                if isinstance(item.block_ref, PropertyBlock):
                    if item.block_ref.ref_block.title == 'Input':
                        self.gr_scene_ref.scene_ref.remove_in_prop()
                    else:
                        self.gr_scene_ref.scene_ref.remove_out_prop()

    def wheelEvent(self, event: 'QtGui.QWheelEvent') -> None:
        """
        Override the event to enable zoom

        """

        factor = 1 / dim.ZOOM_IN_FACTOR

        if event.angleDelta().y() > 0:
            factor = dim.ZOOM_IN_FACTOR
            self.zoom += dim.ZOOM_STEP
        else:
            self.zoom -= dim.ZOOM_STEP

        self.set_scale(factor)
