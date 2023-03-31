"""
Module edge.py

This module contains the class Edge for connecting blocks

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from enum import Enum

from coconet.model.block import PropertyBlock


class EdgeType(Enum):
    """
    Different edge styles

    """

    DIRECT_EDGE = 1
    BEZIER_EDGE = 2


class GraphicsEdgeFactory:
    @staticmethod
    def create_edge(edge_type: EdgeType):
        pass


class Edge:
    def __init__(self, scene: 'Scene', start_block: 'Block', end_block: 'Block', edge_type=EdgeType.BEZIER_EDGE):
        # Reference to the scene
        self.scene_ref = scene

        self.view_dim = True
        if isinstance(end_block, PropertyBlock):
            self.view_dim = False

        # Link to sockets
        self.start_skt = start_block.output_sockets[0]
        self.end_skt = end_block.input_sockets[0]

        if len(start_block.output_sockets) == 0:
            self.start_skt = end_block.output_sockets[0]
            self.end_skt = start_block.input_sockets[0]

        self.start_skt.edge = self
        self.end_skt.edge = self

        # Create graphics edge
        self.graphics_edge = GraphicsEdgeFactory().create_edge(edge_type)
        self.update_pos()
        self.scene_ref.graphics_scene.addItem(self.graphics_edge)

    def update_pos(self):
        pass
