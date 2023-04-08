"""
Module project.py

This module contains the Project class for handling pynever's representation and I/O interfaces

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from pynever.networks import SequentialNetwork
from pynever.nodes import LayerNode

import coconet.utils.rep as rep
from coconet.utils.node_wrapper import NodeFactory


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

        self.set_modified(False)

        # File name is stored as a tuple (name, extension)
        if filename is not None:
            self.filename = filename
            self.open()
        else:
            self.filename = ('', '')

    def set_modified(self, value: bool):
        self.scene_ref.editor_widget_ref.main_wnd_ref.setWindowModified(value)

    def is_modified(self) -> bool:
        return self.scene_ref.editor_widget_ref.main_wnd_ref.isWindowModified() and self.nn.nodes

    def delete_last_node(self) -> LayerNode:
        self.set_modified(True)
        return self.nn.delete_last_node()

    def add_to_nn(self, layer_name: str, layer_id: str, data: dict) -> LayerNode:
        """
        This method creates the corresponding layer node to the graphical block
        and updates the network

        Parameters
        ----------
        layer_name : str
            The LayerNode name
        layer_id : str
            The id to assign to the new node
        data : dict
            The parameters of the node

        Returns
        ----------
        LayerNode
            The node added to the network

        """

        new_node = NodeFactory.create_node(layer_name, layer_id, data, self.last_out_dim())
        self.nn.add_node(new_node)
        self.set_modified(True)

        return new_node

    def link_to_nn(self, node: LayerNode):
        """
        Alternative method for adding a layer directly

        Parameters
        ----------
        node : LayerNode
            The node to add directly

        """

        self.nn.add_node(node)
        self.set_modified(True)

    def last_out_dim(self) -> tuple:
        """
        Compute and return the last node out_dim if there are nodes already,
        read from the input block otherwise

        Returns
        ----------
        tuple
            The last output dimension

        """

        if self.nn.is_empty():
            return rep.text2tuple(self.scene_ref.input_block.attr_dict['Dimension'][1])
        else:
            return self.nn.get_last_node().out_dim

    def open(self):
        pass
