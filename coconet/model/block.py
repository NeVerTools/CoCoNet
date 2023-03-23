"""
Module block.py

This module contains the logic class Block and its children LayerBlock, FunctionalBlock and PropertyBlock

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from uuid import uuid4


class Block:
    def __init__(self, scene: 'Scene'):
        # Reference to the scene
        self.scene_ref = scene

        # Block identifier
        self.block_id = uuid4()


class FunctionalBlock(Block):
    def __init__(self, scene: 'Scene', is_input: bool = True):
        super().__init__(scene)
        pass
