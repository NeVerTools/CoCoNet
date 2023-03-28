"""
Module scene.py

This module contains the class Scene which is used as the manager of logic and graphics objects.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from coconet.model.block import FunctionalBlock, Block
from coconet.model.project import Project
from coconet.view.graphics_scene import GraphicsScene
from coconet.view.graphics_view import GraphicsView


class Scene:
    """
    This class is the manager of logic and graphics objects. It has connections to the
    GraphicsScene class which handles the graphics objects, the GraphicsView class which
    renders them and the Project class which handles the pynever representation of the NN

    """

    def __init__(self, editor_widget: 'CoCoNetWidget'):
        # Reference to the editor widget
        self.editor_widget_ref = editor_widget

        # Dictionary of the displayed blocks
        self.blocks = {}

        # Sequential order of the blocks
        self.sequential_list = []

        # Blocks counter
        self.blocks_number = 0

        # Default distance between blocks
        self.block_distance = 100

        # Initialize I/O blocks
        self.input_block, self.output_block = self.init_io()

        # Input property block
        self.pre_block = None

        # Output property block
        self.post_block = None

        # Project with pynever NN object and interfaces
        self.project = Project(self)

        # Graphics scene
        self.graphics_scene = GraphicsScene(self)

        # Graphics view
        self.view = GraphicsView(self.graphics_scene)
        self.editor_widget_ref.splitter.addWidget(self.view)

    def init_io(self) -> (Block, Block):
        """
        This method creates the input and output blocks, which are permanent

        Returns
        ----------
        tuple
            Input block and Output block

        """

        input_block = FunctionalBlock(self, True)
        output_block = FunctionalBlock(self, False)

        # Add to blocks dict
        self.blocks[input_block.id] = input_block
        self.blocks[output_block.id] = output_block

        # Init start position in the view
        # input_block.setPos(-300, -60)
        # output_block.setPos(100, -60)

        self.blocks_number += 2

        return input_block, output_block

    def add_block(self, block_data: dict, block_sign: str, loading_dict: dict = None, block_id=None):
        pass
