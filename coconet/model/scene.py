"""
Module scene.py

This module contains the class Scene which is used as the manager of logic and graphics objects.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
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

        # Project with pynever NN object and interfaces
        self.project = Project(self)

        # Dictionary of the displayed blocks
        self.blocks = {}

        # Blocks counter
        self.blocks_number = 0

        # Default distance between blocks
        self.block_distance = 100

        # Input property block
        self.p_input_block = None

        # Output property block
        self.p_output_block = None

        # Graphics scene
        self.graphics_scene = GraphicsScene(self)

        # Graphics view
        self.view = GraphicsView(self.graphics_scene)
        self.editor_widget_ref.add_to_layout(self.view)
