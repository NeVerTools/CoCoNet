"""
Module block.py

This module contains the logic class Block and its children LayerBlock, FunctionalBlock and PropertyBlock

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from typing import Optional
from uuid import uuid4

import coconet.resources.styling.dimension as dim
from coconet.model.socket import Socket, SocketPosition, SocketType
from coconet.view.components.graphics_block import GraphicsBlock, BlockContentWidget


class Block:
    """
    This class is the logic representation of a Block. It stores the block data
    and references the graphics representation in the GraphicsBlock class.

    Attributes
    ----------
    scene_ref : Scene
        Reference to the scene

    _id : str
        Identifier of the block

    title : str
        Title of the block in the editor

    attr_dict : dict
        Dictionary containing all the block parameters

    input_sockets : list
        Sockets for input connection

    output_sockets : list
        Sockets for output connection

    graphics_block : GraphicsBlock
        Graphics block representation of this object

    is_newline : bool
        Flag used when rendering large networks

    """

    def __init__(self, scene: 'Scene'):
        # Reference to the scene
        self.scene_ref = scene

        # Block identifier
        self._id = str(uuid4())

        # Block title
        self.title = ''

        # Block attributes dict
        self.attr_dict = dict()

        # Sockets
        self.input_sockets = []
        self.output_sockets = []

        # Graphics representation of the block
        self.graphics_block = None

        # Utility attributes
        self.is_newline = False

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def pos(self):
        return self.graphics_block.pos()

    @property
    def width(self):
        return self.graphics_block.width

    @property
    def height(self):
        return self.graphics_block.height

    def has_parameters(self) -> bool:
        return len(self.attr_dict['parameters']) > 0

    def has_input(self) -> bool:
        if self.input_sockets:
            return self.input_sockets[-1].has_edge()

    def has_output(self) -> bool:
        if self.output_sockets:
            return self.output_sockets[-1].has_edge()

    def init_sockets(self, inputs, outputs):
        """
        Draw the sockets given the inputs and outputs of the block

        """

        for i in range(len(inputs)):
            socket = Socket(self, i, SocketPosition.LEFT_TOP, SocketType.INPUT)
            self.input_sockets.append(socket)

        for i in range(len(outputs)):
            socket = Socket(self, i, SocketPosition.RIGHT_TOP, SocketType.OUTPUT)
            self.output_sockets.append(socket)

    def get_socket_pos(self, index: int, position: SocketPosition, absolute: bool) -> tuple:
        """
        Method to return the relative or absolute position of the sockets of this block

        """

        if position == SocketPosition.LEFT_TOP:
            x = 0
        else:
            x = self.graphics_block.width

        y = dim.TITLE_HEIGHT + dim.TITLE_PAD + dim.EDGE_ROUNDNESS + index * dim.SOCKET_SPACING

        if absolute:
            return x + self.pos.x(), y + self.pos.y()
        else:
            return x, y

    def previous(self) -> Optional['Block']:
        """
        Utility method to retrieve the previous block

        Returns
        ----------
        The previous block, if it exists, None otherwise

        """

        prev = None

        if self.has_input():
            prev_block_idx = self.scene_ref.sequential_list.index(self.id) - 1
            prev_block_id = self.scene_ref.sequential_list[prev_block_idx]
            prev = self.scene_ref.blocks[prev_block_id]

        return prev

    def set_rel_to(self, other: 'Block'):
        """
        Utility method to set the block position with respect to another one

        Parameters
        ----------
        other : Block
            The relative block

        """

        if self.scene_ref.input_block is not None:
            if self.previous() == self.scene_ref.input_block:
                self.is_newline = True
        else:
            self.is_newline = False

        new_pos_width = other.pos.x() + other.width + self.scene_ref.block_distance

        if other.height != self.height:
            new_pos_height = other.pos.y() - self.height / 2 + other.height / 2
        else:
            new_pos_height = other.pos.y()

        if new_pos_width > dim.SCENE_BORDER:
            self.is_newline = True

            # Newline
            prev = self.previous()
            highest = prev

            while not prev.is_newline:
                prev = prev.previous()
                if prev.height > highest.height:
                    highest = prev

            new_pos_width = prev.pos.x()
            new_pos_height = highest.pos.y() + highest.height + dim.NEXT_BLOCK_DISTANCE

        self.graphics_block.setPos(new_pos_width, new_pos_height)

        # Move the output block
        end_pos_width = self.pos.x() + self.width + self.scene_ref.block_distance

        if self.height != self.scene_ref.output_block:
            end_pos_height = self.pos.y() - self.scene_ref.output_block.height / 2 + self.height / 2
        else:
            end_pos_height = other.pos.y()

        self.scene_ref.output_block.graphics_block.setPos(end_pos_width, end_pos_height)

        self.update_edges()

    def update_edges(self):
        """
        This method updates the edges connected to the block

        """

        for socket in self.input_sockets + self.output_sockets:
            if socket.edge is not None:
                socket.edge.update_pos()


class LayerBlock(Block):
    def __init__(self, scene: 'Scene', inputs: list, outputs: list, build_dict: dict,
                 key_signature: str = '', block_id: str = None):
        super().__init__(scene)

        # Signature for dictionary map
        self.signature = 'blocks:' + key_signature
        self.title = self.signature.split(':')[-1]

        # If an ID is provided use it
        if block_id is not None:
            self._id = block_id

        # Init graphics block
        self.graphics_block = GraphicsBlock(self)

        # Copy parameters and create block layout
        self.attr_dict = build_dict
        if self.has_parameters():
            self.graphics_block.content = BlockContentWidget(self, self.attr_dict)

        self.scene_ref.graphics_scene.addItem(self.graphics_block)

        self.init_sockets(inputs, outputs)

    def remove(self):
        """
        Procedure to remove this block

        """

        end_block = self.scene_ref.blocks['END']
        end_pos_width = self.pos.x()

        if self.height != end_block.height:
            end_pos_height = self.pos.y() - end_block.height / 2 + self.height / 2
        else:
            end_pos_height = self.pos.y()

        end_block.graphics_block.setPos(end_pos_width, end_pos_height)

        # Remove connected edges
        for socket in self.input_sockets + self.output_sockets:
            if socket.edge is not None:
                socket.edge.remove()
                socket.edge = None

        # Remove from graphics scene
        self.scene_ref.graphics_scene.removeItem(self.graphics_block)
        self.graphics_block = None
        del self


class FunctionalBlock(Block):
    def __init__(self, scene: 'Scene', is_input: bool = True):
        super().__init__(scene)

        # I/O data
        if is_input:
            self.title = 'Input'
            sockets = [[], [1]]

        else:
            self.title = 'Output'
            sockets = [[1], []]

        # Id from data
        self._id = self.scene_ref.editor_widget_ref.functional_data[self.title]['id']
        self.signature = 'functional:' + self.title

        self.attr_dict = self.scene_ref.editor_widget_ref.functional_data[self.title]

        # Init content
        self.graphics_block = GraphicsBlock(self)
        self.graphics_block.content = BlockContentWidget(self)
        self.scene_ref.graphics_scene.addItem(self.graphics_block)

        self.init_sockets(*sockets)


class PropertyBlock(Block):
    pass
