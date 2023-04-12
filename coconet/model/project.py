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

    filename : (str, str)
        The filename of the network stored in a tuple (name, extension)
    
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

    def is_modified(self) -> bool:
        return self.scene_ref.editor_widget_ref.main_wnd_ref.isWindowModified() and self.nn.nodes

    def set_modified(self, value: bool):
        self.scene_ref.editor_widget_ref.main_wnd_ref.setWindowModified(value)

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
            return rep.text2tuple(self.scene_ref.input_block.content.wdg_param_dict['Dimension'][1])
        else:
            return self.nn.get_last_node().out_dim

    def reset_nn(self, new_input_id: str, caller_id: str):
        """
        If a functional block is updated, the network is re-initialized

        Parameters
        ----------
        new_input_id : str
            New identifier for the network input
        caller_id : str
            The block that was updated (either 'INP' or 'END')

        """

        if caller_id == 'INP':
            if self.nn.input_id != new_input_id:
                self.nn = SequentialNetwork('net', new_input_id)

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

        new_node = NodeFactory.create_layernode(layer_name, layer_id, data, self.last_out_dim())
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

    def refresh_node(self, node_id: str, params: dict):
        """
        This method propagates the visual modifications to the logic node
        by deleting and re-adding it to the network

        Parameters
        ----------
        node_id : str
            The id key to the nodes dictionary
        params : dict
            The node parameters

        """

        # Delete and re-create the node
        to_remove = self.nn.nodes[node_id]
        self.delete_last_node()

        data = rep.format_data(params)
        new_node = self.add_to_nn(str(to_remove.__class__.__name__), node_id, data)

        # Update dimensions
        dim_wdg = self.scene_ref.output_block.content.wdg_param_dict['Dimension'][0]
        dim_wdg.setText(str(new_node.out_dim))
        self.scene_ref.output_block.content.wdg_param_dict['Dimension'][1] = new_node.out_dim

    def delete_last_node(self) -> LayerNode:
        self.set_modified(True)
        return self.nn.delete_last_node()

    def open(self):
        pass
