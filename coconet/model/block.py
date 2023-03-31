"""
Module block.py

This module contains the logic class Block and its children LayerBlock, FunctionalBlock and PropertyBlock

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
from typing import Optional
from uuid import uuid4

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
        self._id = uuid4()

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

    def set_rel_to(self, prev_block: 'Block'):
        pass

    def has_input(self) -> bool:
        if self.input_sockets:
            return self.input_sockets[-1].has_edge()


class LayerBlock(Block):
    def __init__(self, scene: 'Scene', inputs: list, outputs: list, build_dict: dict,
                 key_signature: str = '', block_id: str = None):
        super().__init__(scene)

        # Signature for dictionary map
        self.signature = key_signature
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
