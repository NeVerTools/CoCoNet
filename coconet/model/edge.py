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


class Edge:
    def __init__(self, scene: 'Scene', start_block: 'Block', end_block: 'Block', type=EdgeType.BEZIER_EDGE):
        # Reference to the scene
        self.scene_ref = scene

        self.view_dim = True
        if isinstance(end_block, PropertyBlock):
            self.view_dim = False
