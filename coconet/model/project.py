"""
Module project.py

This module contains the Project class for handling pynever's representation and I/O interfaces

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""


class Project:
    def __init__(self, scene: 'Scene'):
        # Reference to the scene
        self.scene_ref = scene
