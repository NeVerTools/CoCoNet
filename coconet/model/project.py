"""
Module project.py

This module contains the Project class for handling pynever's representation and I/O interfaces

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from pynever.networks import SequentialNetwork


class Project:
    """
    This class serves as a manager for the definition of a pynever Neural Network object.
    It provides methods to update the network reflecting the actions in the graphical interface
    
    Attributes
    ----------
    scene_ref : Scene
        Reference to the scene
        
    nn : SequentialNetwork
        The network object instantiated by the interface
    
    """
    
    def __init__(self, scene: 'Scene', filename: str = None):
        # Reference to the scene
        self.scene_ref = scene

        # Default init is sequential, future extensions should either consider multiple initialization or
        # on-the-fly switch between Sequential and ResNet etc.
        self.nn = SequentialNetwork('net', self.scene_ref.input_block.content.wdg_param_dict['Name'][1])
        self.init_new()
        
        self.set_modified(False)
        
        # File name is stored as a tuple (name, extension)
        if filename is not None:
            self.filename = filename
            self.open()
        else:
            self.filename = ('', '')

    def set_modified(self, value: bool):
        self.scene_ref.editor_widget_ref.main_wnd_ref.setWindowModified(value)

    def init_new(self):
        pass

    def open(self):
        pass
