"""
Module block.py

This module contains the logic class Block and its children LayerBlock, FunctionalBlock and PropertyBlock

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from uuid import uuid4

from coconet.view.ui.graphics_block import GraphicsBlock, BlockContentWidget


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

    graphics_block : GraphicsBlock
        Graphics block representation of this object

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

        # Graphics representation of the block
        self.graphics_block = None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    def has_parameters(self) -> bool:
        return len(self.attr_dict['parameters']) > 0


class LayerBlock(Block):
    def __init__(self, scene: 'Scene', inputs: list, outputs: list, build_dict: dict,
                 key_signature: str = '', block_id: str = None):
        super().__init__(scene)

        # Signature for dictionary map
        self._signature = key_signature
        self.title = self._signature.split(':')[-1]

        # If an ID is provided use it
        if block_id is not None:
            self._id = block_id

        # Init graphics block
        self.graphics_block = GraphicsBlock(self)
        self.scene_ref.graphics_scene.addItem(self.graphics_block)

        # Copy parameters and create block layout
        self.attr_dict = build_dict
        if self.has_parameters():
            self.graphics_block.set_content(BlockContentWidget(self, self.attr_dict))


class FunctionalBlock(Block):
    def __init__(self, scene: 'Scene', is_input: bool = True):
        super().__init__(scene)
        pass
