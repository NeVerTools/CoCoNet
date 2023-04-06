"""
Module project.py

This module contains the Project class for handling pynever's representation and I/O interfaces

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from pynever.networks import SequentialNetwork


class Project:
    def __init__(self, scene: 'Scene'):
        # Reference to the scene
        self.scene_ref = scene

        # Default init is sequential, future extensions should either consider multiple initialization or
        # on-the-fly switch between Sequential and ResNet etc.
        self.nn = SequentialNetwork('net', self.scene_ref.input_block.content.wdg_param_dict['Name'][1])
