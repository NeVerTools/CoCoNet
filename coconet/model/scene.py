"""
Module scene.py

This module contains the class Scene which is used as the manager of logic and graphics objects.

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from typing import Optional, Iterable

import coconet.utils.repr as u
from coconet.model.block import FunctionalBlock, Block, LayerBlock, PropertyBlock
from coconet.model.edge import Edge
from coconet.model.project import Project
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

    @staticmethod
    def get_variables(f_block: FunctionalBlock) -> Iterable:
        return u.create_variables_from(f_block.attr_dict['name'][1],
                                       u.text2tuple(f_block.attr_dict['dimension'][1]))

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
        input_block.graphics_block.setPos(-300, -60)
        output_block.graphics_block.setPos(100, -60)

        self.blocks_count += 2

        return input_block, output_block

    def add_layer_block(self, block_data: dict, block_sign: str, block_id=None) -> Optional[LayerBlock]:
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

        # Add the block in the list and dictionary
        self.sequential_list.insert(len(self.sequential_list) - 1, added_block.id)
        self.blocks[added_block.id] = added_block

        # Remove last edge
        last_block_socket = self.output_block.input_sockets[0]
        if last_block_socket.has_edge():
            last_block_socket.get_edge().remove()

        # Add two new edges
        self.add_edge(prev, added_block)
        self.add_edge(added_block, self.output_block)

        # Set position
        added_block.set_rel_to(prev)

        # Manage network
        # TODO
        # self.update_out_dim()
        # self.update_edge_label(added_block)

        self.view.centerOn(added_block.pos.x() + added_block.width / 2, added_block.pos.y() + added_block.height / 2)

        return added_block

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

    def create_property(self, name: str, parent: FunctionalBlock, prop_cnt: PropertyContainer = None):
        """
        This function defines a property given the input or output block

        Parameters
        ----------
        name
        parent
        prop_cnt

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
                        has_edits = self.edit_property(new_block)
                    else:
                        has_edits = True
                        new_block.smt_string = prop_cnt.smt_string
                        new_block.variables = prop_cnt.variables

                    if has_edits:
                        new_block.draw_property()

                        if parent.title == 'Input' and not self.pre_block:
                            self.pre_block = new_block
                        elif parent.title == 'Output' and not self.post_block:
                            self.post_block = new_block
        else:
            dialog = MessageDialog('No network defined for adding a property', MessageType.ERROR)
            dialog.exec()

    def edit_property(self, block: PropertyBlock) -> bool:
        pass

    def remove_in_prop(self):
        pass

    def remove_out_prop(self):
        pass
