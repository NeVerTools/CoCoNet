"""
Module scene.py

This module contains the class Scene which is used as the manager of logic and graphics objects.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QGraphicsItem
from pynever.nodes import LayerNode

import coconet.utils.rep as rep
from coconet.model.component.block import FunctionalBlock, Block, LayerBlock, PropertyBlock
from coconet.model.component.edge import Edge
from coconet.model.project import Project
from coconet.resources.styling.custom import CustomLabel
from coconet.utils.container import PropertyContainer
from coconet.view.graphics_scene import GraphicsScene
from coconet.view.graphics_view import GraphicsView
from coconet.view.ui.dialog import ConfirmDialog, MessageDialog, MessageType


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
        self.blocks_count = 0

        # Graphics scene
        self.graphics_scene = GraphicsScene(self)

        # Graphics view
        self.view = GraphicsView(self.graphics_scene)
        self.editor_widget_ref.splitter.addWidget(self.view)

        # Initialize I/O blocks
        self.input_block, self.output_block = self.init_io()

        # Input property block
        self.pre_block = None

        # Output property block
        self.post_block = None

        # Project with pynever NN object and interfaces
        self.project = Project(self)

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

        # Add to sequential list
        self.sequential_list.append(input_block.id)
        self.sequential_list.append(output_block.id)

        # Init start position in the view
        input_block.graphics_block.setPos(-300, -60)
        output_block.graphics_block.setPos(100, -60)

        self.blocks_count += 2

        return input_block, output_block

    def has_network(self) -> bool:
        return False if self.project is None else self.project.nn.is_empty()

    def has_properties(self) -> bool:
        return self.pre_block is not None or self.post_block is not None

    def load_properties(self, prop_dict: dict):
        """
        Load existing properties from a dictionary <ID> : <PropertyFormat>

        """

        if len(prop_dict.keys()) <= 2:  # At most one pre-condition and one post-condition
            available_list = [self.input_block.tile, self.output_block.title,
                              self.project.nn.get_last_node().identifier]

            # Check variables compatibility
            for prop_id in prop_dict.keys():
                if prop_id not in available_list:
                    raise Exception('This property appears to be defined on another network!\n'
                                    f'Unknown variable: {prop_id}')

            # Check output id
            if self.project.nn.get_last_node().identifier in prop_dict.keys():
                new_smt_string = prop_dict[self.project.nn.get_last_node().identifier].smt_string.replace(
                    self.project.nn.get_last_node().identifier, self.output_block.title)

                new_variable_list = []
                for variable in prop_dict[self.project.nn.get_last_node().identifier].variables:
                    new_variable_list.append(variable.replace(
                        self.project.nn.get_last_node().identifier, self.output_block.title))

                prop_dict[self.output_block.title] = PropertyContainer(new_smt_string, new_variable_list)
                prop_dict.pop(self.project.nn.get_last_node().identifier)

            for k, v in prop_dict.keys():
                if k == self.input_block.tile:
                    self.add_property_block('Generic SMT', self.input_block, v)
                elif k == self.output_block.title or k == self.project.nn.get_last_node().identifier:
                    self.add_property_block('Generic SMT', self.output_block, v)

    def add_layer_block(self, block_data: dict, block_sign: str,
                        block_id: str = None, load_dict: dict = None) -> Optional[LayerBlock]:
        """
        This method adds a layer block in the Scene and draws it in the View

        Parameters
        ----------
        block_data : dict
            Block info stored in dictionary

        block_sign : str
            Signature of the block

        block_id : str, Optional
            Identifier provided when reading a network file

        load_dict : dict, Optional
            Extra dictionary with values for loading from file

        Returns
        ----------
        Optional[LayerBlock]
            Returns the LayerBlock object if created, None otherwise

        """

        # Check if there is an output property
        if self.post_block:
            dialog = ConfirmDialog('Confirm required', 'If you edit the network the output property will be removed\n'
                                                       'Do you wish to proceed?')
            dialog.exec()

            if dialog.confirm:
                self.remove_out_prop()
            else:
                return None

        # Add the block
        added_block = LayerBlock(self, [1], [1], block_data, block_sign, block_id)
        self.blocks_count += 1

        # Ex last block is the second last (output is included)
        prev = self.blocks.get(self.sequential_list[-2])
        if prev.has_parameters():
            prev.graphics_block.content.toggle_content_enabled(False)

        # Add the block in the list and dictionary
        self.sequential_list.insert(len(self.sequential_list) - 1, added_block.id)
        self.blocks[added_block.id] = added_block

        # Remove last edge
        last_block_socket = self.output_block.input_sockets[0]
        if last_block_socket.edge is not None:
            last_block_socket.edge.remove()

        # Add two new edges
        self.add_edge(prev, added_block)
        self.add_edge(added_block, self.output_block)

        # Set position
        added_block.set_rel_to(prev)

        # Case 1: the network is loaded from file
        if load_dict is not None and block_id is not None:
            if hasattr(added_block.graphics_block, 'content'):
                added_node = self.project.nn.nodes[block_id]
                self.update_block_params(added_block, added_node)
        # Case 2: the network is built on the fly
        else:
            try:
                if hasattr(added_block.graphics_block, 'content'):
                    added_node = self.project.add_to_nn(added_block.attr_dict['name'],
                                                        added_block.id,
                                                        rep.format_data(added_block.attr_dict))
                    self.update_block_params(added_block, added_node)
                else:
                    self.project.add_to_nn(added_block.attr_dict['name'],
                                           added_block.id, {})
            except Exception as e:
                dialog = MessageDialog(str(e), MessageType.ERROR)
                dialog.exec()

                self.remove_block(added_block, logic=False)
                return None

        self.update_out_dim()
        self.update_edge_dim(added_block)

        self.view.centerOn(added_block.pos.x() + added_block.width / 2, added_block.pos.y() + added_block.height / 2)

        return added_block

    def add_property_block(self, name: str, parent: FunctionalBlock, prop_cnt: PropertyContainer = None):
        """
        This function defines a property given the input or output block

        Parameters
        ----------
        name : str
            The name of the property

        parent : FunctionalBlock
            The block to attach the property to

        prop_cnt : PropertyContainer
            The container object for property data

        """

        if len(self.blocks) > 2:  # Check there are layers in the network
            new_block = PropertyBlock(self, name, parent)

            # Check there are no properties already
            if parent.get_property_block() is not None:
                dialog = ConfirmDialog('Replace property',
                                       'There is a property already\nReplace it?')
                dialog.exec()

                if dialog.confirm:
                    if parent.title == 'Input':
                        self.remove_in_prop()
                    else:
                        self.remove_out_prop()

                    if prop_cnt is None:
                        has_edits = new_block.edit()
                    else:
                        has_edits = True
                        new_block.smt_string = prop_cnt.smt_string
                        new_block.variables = prop_cnt.variables

                    if has_edits:
                        new_block.draw()

                        if parent.title == 'Input' and not self.pre_block:
                            self.pre_block = new_block
                        elif parent.title == 'Output' and not self.post_block:
                            self.post_block = new_block
        else:
            dialog = MessageDialog('No network defined for adding a property', MessageType.ERROR)
            dialog.exec()

    def add_edge(self, prev: Block, cur: Block):
        """
        Add and draw the edge connecting two blocks

        Parameters
        ----------
        prev : Block
            The block from where the edge starts

        cur : Block
            The block where the edge ends

        """

        if self.blocks_count > 0:
            return Edge(self, prev, cur)

    @staticmethod
    def update_block_params(added_block: LayerBlock, added_node: LayerNode):
        """
        Display the correct parameters on the graphics block

        Parameters
        ----------
        added_block : LayerBlock
            The block displayed in the view

        added_node : LayerNode
            The corresponding neural network node

        """

        if hasattr(added_block.content, 'wdg_param_dict'):
            for param_name, param_value in added_block.content.wdg_param_dict.items():
                q_wdg = param_value[0]
                if isinstance(q_wdg, CustomLabel):
                    if hasattr(added_node, param_name):
                        node_param = getattr(added_node, param_name)
                        if isinstance(node_param, np.ndarray):
                            sh = node_param.shape
                            q_wdg.setText(rep.tuple2text(sh))
                        else:
                            q_wdg.setText(str(node_param))

    def update_edges(self):
        """
        Add new edges and reposition after an update occurs

        """

        # Blocks to connect
        start_block = None
        end_block = None

        for _, block in self.blocks.items():
            if not block.has_input():
                end_block = block
            elif not block.has_output():
                start_block = block

        if start_block is not None and end_block is not None:
            if start_block is not self.input_block or end_block is not self.output_block:
                self.add_edge(start_block, end_block)

    def update_edge_dim(self, block: Block):
        """
        Display the output dimension of a block in the following edge

        """

        if block.has_input():
            prev_id = block.input_sockets[0].edge.start_skt.block_ref.id

            if prev_id != 'INP':
                prev_out_dim = self.project.nn.nodes[prev_id].out_dim
                block.input_sockets[0].edge.update_label(rep.tuple2text(prev_out_dim))

    def update_out_dim(self):
        """
        Write the output dimension in the output block

        """

        last_id = self.sequential_list[-2]
        dim_wdg = self.output_block.content.wdg_param_dict['Dimension'][0]

        if last_id == 'INP':
            dim_value = ''
        else:
            last_node = self.project.nn.get_last_node()
            dim_value = rep.tuple2text(last_node.out_dim, prod=False)

        dim_wdg.setText(dim_value)
        self.output_block.content.wdg_param_dict['Dimension'][1] = dim_value

    def remove_block(self, block: Block, logic: bool = False):
        """
        Remove a block both from the view and from the network

        Parameters
        ----------
        block : Block
            The block to delete

        logic : bool, Optional
            Flag for deleting the corresponding node in the network

        """

        # If the block is a property logic is forced to False
        if isinstance(block, PropertyBlock):
            logic = False
        elif self.post_block is not None:
            dialog = ConfirmDialog('Confirm required', 'If you edit the network the output property will be removed\n'
                                                       'Do you wish to proceed?')
            dialog.exec()

            if dialog.confirm:
                self.remove_out_prop()
            else:
                return  # Early stopping

        if not isinstance(block, FunctionalBlock):
            ref_id = block.id
            block.remove()

            if ref_id in self.blocks:
                self.blocks.pop(ref_id)
                self.sequential_list.remove(ref_id)
                self.blocks_count -= 1
                self.update_out_dim()

                # Re-enable widgets in the previous block
                prev_block = self.blocks[self.sequential_list[-2]]
                if prev_block.has_parameters():
                    prev_block.graphics_block.content.toggle_content_enabled(True)

            elif self.pre_block is not None and ref_id == self.pre_block.id:
                self.pre_block = None
            elif self.post_block is not None and ref_id == self.post_block.id:
                self.post_block = None

            self.update_edges()

            if logic:
                self.project.delete_last_node()

    def remove_in_prop(self):
        if self.pre_block is not None:
            self.pre_block.remove()
            self.pre_block = None

    def remove_out_prop(self):
        if self.post_block is not None:
            self.post_block.remove()
            self.post_block = None

    def clear_scene(self):
        """
        Clear all graphics objects, dictionaries and re-init the I/O blocks

        """

        self.graphics_scene.clear()
        self.blocks = {}
        self.sequential_list = []
        self.blocks_count = 0
        self.pre_block = None
        self.post_block = None
        self.input_block, self.output_block = self.init_io()

    def get_item_at(self, pos: 'QPointF') -> 'QGraphicsItem':
        return self.view.itemAt(pos)
