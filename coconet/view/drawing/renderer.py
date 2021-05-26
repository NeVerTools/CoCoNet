import copy

from PyQt5.QtCore import pyqtSignal

from coconet.core.controller.nodewrapper import NodeOps
from coconet.core.controller.project import Project
from coconet.core.controller.pynevertemp.networks import NeuralNetwork
from coconet.view.drawing.element import Block


class SequentialNetworkRenderer:
    """ This class is an attribute of the Canvas object: reading its Project
    attribute, it builds a list of related GraphicBlockObjects composing the
    graph and changing it according to user events.

    Attributes
    ----------
    project: Project
    disconnected_network: dict
        Dictionary containing bot connected and disconnected blocks

    Methods
    ----------
    is_sequential(bool):
        Returns true if the network is sequential
    add_single_node(str):
        Adds to the network a block without connections
    add_new_block(Block):
        explained below
    add_edge(str, str):
        explained below
    delete_edge(str, SplitMode):
        explained below
    edit_node(str, dict):
        explained below
    update_network_from(str, str):
        explained below
    insert_node(str, str):
        explained below
    draw_network(NeuralNetwork):
        explained below
    has_nodes_before(str):
        explained bel

    """

    drawable_block = pyqtSignal()

    def __init__(self, project: Project, blocks):
        self.project = project
        self.blocks = blocks
        # dictionary containing all the nodes in the project, including the
        # disconnected ones
        self.disconnected_network = dict()

    def is_sequential(self) -> bool:
        """
        This method controls if the network is sequential.
       :return: bool
        """
        net = self.project.NN
        if len(self.disconnected_network) < 2:
            # If there is only one node, the network is sequential
            return True
        else:
            # Each node must lead to his next with one edge, until the
            # last block
            n = net.get_first_node().identifier
            n_nodes = 1
            while len(net.edges[n]) == 1:
                n = net.get_next_node(net.nodes[n]).identifier
                n_nodes += 1

            # If one node has one or more edges, the net is not sequential
            if len(net.edges[n]) > 1:
                return False
            # If reaching the last node all nodes have not been counted,
            # the net is not sequential
            elif n_nodes is not len(net.nodes):
                return False
            else:
                return True

    def add_single_node(self, node_id: str):
        """
        This method adds a new node to the network without connections
       :param node_id: str
        """
        try:
            if node_id in self.disconnected_network:
                block = self.disconnected_network[node_id]
                node = NodeOps.create_node(block.node.class_name, node_id,
                                           block.block_data, block.in_dim)
                self.project.NN.nodes[node_id] = node
                self.project.NN.edges[node_id] = []
        except Exception as e:
            raise e

    def add_new_block(self, block: Block):
        """
        This method takes a new block added by the user creating a LayerNode
        object
        """
        self.disconnected_network[block.block_id] = block

    def add_edge(self, origin_id, destination_id):
        """
        This method adds a connection between two nodes, only if the network
        remains sequential.
       :param origin_id: str
       :param destination_id: str
        """
        origin_block = self.disconnected_network[origin_id]
        destination_block = self.disconnected_network[destination_id]

        # If there are not edges yet, the origin becomes the first node and
        # then the second is added
        if not self.project.NN.nodes:
            st_node = NodeOps.create_node(origin_block.node.class_name,
                                          origin_block.block_id,
                                          origin_block.block_data,
                                          origin_block.in_dim)
            nd_node = NodeOps.create_node(destination_block.node.class_name,
                                          destination_block.block_id,
                                          destination_block.block_data,
                                          st_node.out_dim)
            self.project.NN.add_node(st_node)
            self.project.NN.add_node(nd_node)

            self.disconnected_network[nd_node.identifier].block_data = NodeOps.node2data(nd_node)[0]
            self.disconnected_network[nd_node.identifier].in_dim = nd_node.in_dim
            self.disconnected_network[nd_node.identifier].update_labels()

            return True
        elif origin_id not in self.project.NN.nodes and \
                destination_id not in self.project.NN.nodes:
            # If the network is not empty but the connection to draw is
            # between two new nodes, the connection is drawn,
            # but probably the network will not be saved
            st_node = NodeOps.create_node(origin_block.node.class_name,
                                          origin_block.block_id,
                                          origin_block.block_data,
                                          origin_block.in_dim)
            nd_node = NodeOps.create_node(destination_block.node.class_name,
                                          destination_block.block_id,
                                          destination_block.block_data,
                                          st_node.out_dim)

            self.project.NN.nodes[origin_id] = st_node
            self.project.NN.nodes[destination_id] = nd_node
            self.project.NN.edges[origin_id] = [destination_id]
            self.project.NN.edges[destination_id] = []

            self.disconnected_network[nd_node.identifier].block_data = NodeOps.node2data(nd_node)[0]
            self.disconnected_network[nd_node.identifier].in_dim = nd_node.in_dim
            self.disconnected_network[nd_node.identifier].update_labels()

            return True
        elif len(self.project.NN.nodes) == (self.len_edges() + 1) and \
                origin_id in self.project.NN.nodes and \
                destination_id in self.project.NN.nodes:
            # If the number of nodes in the network is equal to the number of edges +1 and
            # both origin an destination already belong to the net, then the net will be
            # "circular" and it is not possible
            return False
        elif (origin_id not in self.project.NN.nodes or len(self.project.NN.edges[origin_id]) == 0) and \
                not self.has_nodes_before(destination_id):
            # After the origin there are not other nodes, and before the
            # destination there are not other nodes

            # Checking if the two nodes have been already created
            if origin_id not in self.project.NN.nodes:
                orig_node = NodeOps.create_node(origin_block.node.class_name,
                                                origin_block.block_id,
                                                origin_block.block_data,
                                                origin_block.in_dim)
                self.project.NN.nodes[origin_id] = orig_node
                self.project.NN.edges[origin_id] = []
            else:
                orig_node = self.project.NN.nodes[origin_id]

            if destination_id not in self.project.NN.nodes:
                dest_node = NodeOps.create_node(destination_block.node.class_name,
                                                destination_block.block_id,
                                                destination_block.block_data,
                                                orig_node.out_dim)
                self.project.NN.nodes[destination_id] = dest_node
                self.project.NN.edges[destination_id] = []
            else:
                dest_node = self.project.NN.nodes[destination_id]
                NodeOps.update_node_input(dest_node, orig_node.out_dim)
                # If the destination had other blocks after it, they are updated
                if len(self.project.NN.edges[destination_id]) > 0:
                    self.update_network_from(destination_id, origin_id)

            self.project.NN.edges[origin_id].append(destination_id)

            self.disconnected_network[dest_node.identifier].block_data = NodeOps.node2data(dest_node)[0]
            self.disconnected_network[dest_node.identifier].in_dim = dest_node.in_dim
            self.disconnected_network[dest_node.identifier].update_labels()
            return True
        else:
            # else return False because the network is not Sequential
            return False

    def len_edges(self) -> int:
        """
        This method returns the number of edges in the network
       :return: int
        """
        edges = 0
        for edge in self.project.NN.edges.values():
            edges += len(edge)
        return edges

    def has_nodes_before(self, block_id: str) -> bool:
        """
        This method returns true if the passed node has other
        nodes before it.
       :param block_id: str
       :return: bool
        """
        for dest in self.project.NN.edges.values():
            if block_id in dest:
                return True
        return False

    def delete_edge(self, block_id: str):
        """
        This method deletes the edge from the given node.
       :param block_id: str
       :return blocks_to_clean: list
        """
        self.project.NN.edges[block_id] = []

    def edit_node(self, block_id: str, data: dict):
        """
        This method edits the given node, editing the parameters given by
        the dictionary.
       :param block_id: str
       :param data: dict
        """
        # Editing the node depending on its type
        in_dim = None
        if "in_dim" in data.keys():
            in_dim = data["in_dim"]
            data.pop("in_dim")

        # Current data is copied
        old_data = copy.deepcopy(self.disconnected_network[block_id].block_data)
        old_in_dim = copy.deepcopy(self.disconnected_network[block_id].in_dim)

        # New data is updated
        for key, value in data.items():
            self.disconnected_network[block_id].block_data[key] = value
        data = self.disconnected_network[block_id].block_data

        if in_dim is not None:
            self.disconnected_network[block_id].in_dim = in_dim

        try:
            # The network is updated from the edited node
            if block_id in self.project.NN.nodes.keys():
                node = self.project.NN.nodes[block_id]

                # If the input dimension has been modified, it is updated
                if in_dim is not None:
                    NodeOps.update_node_input(node, in_dim)
                NodeOps.update_node_data(node, data)

                if node is not self.project.NN.get_last_node():
                    next_node = self.project.NN.get_next_node(node)
                    self.update_network_from(next_node.identifier, node.identifier)

                return True
            else:
                return False
        except Exception as e:
            # The graphic block is restored
            self.disconnected_network[block_id].block_data = old_data
            self.disconnected_network[block_id].in_dim = old_in_dim
            # The network is restored
            NodeOps.update_node_data(node, old_data)
            if in_dim is not None:
                NodeOps.update_node_input(node, old_in_dim)

            if node is not self.project.NN.get_last_node():
                next_node = self.project.NN.get_next_node(node)
                self.update_network_from(next_node.identifier, node.identifier)

            raise Exception(str(e))

    def delete_node(self, block_id: str):
        """
        This method deletes a node, eventually deleting and creating
        edges to keep the network connected.
       :param block_id: str
        """
        # it will be returned the possible new edge
        new_edge = None
        if block_id in self.project.NN.nodes:
            # Rolling the sequential network to find the previous node
            for id, node in self.project.NN.nodes.items():
                if block_id in self.project.NN.edges[id]:
                    # If the node to remove wasn't the last node, adding the next node
                    # in the edges of the previous node
                    if block_id is not self.project.NN.get_last_node().identifier:
                        next_node_id = self.project.NN.get_next_node(
                            self.project.NN.nodes[block_id]).identifier

                        try:
                            # Update network
                            self.update_network_from(next_node_id, id)

                            # Removing the edge between the previous node and the one to delete
                            self.project.NN.edges[id].remove(block_id)
                            self.project.NN.edges[id].append(next_node_id)
                            new_edge = (id, next_node_id)
                        except Exception as e:
                            raise Exception(e)
                    else:
                        # If the node was the last node
                        self.project.NN.edges[id].remove(block_id)
                    break

            # Deleting the layerNode returning the new edge to draw
            self.project.NN.nodes.pop(block_id)
            self.project.NN.edges.pop(block_id)

            if len(self.project.NN.nodes) == 1:
                only_node = self.project.NN.get_first_node().identifier
                self.project.NN.nodes.pop(only_node)
                self.project.NN.edges.pop(only_node)

        # Deleting the block from the disconnected graph
        self.disconnected_network.pop(block_id)
        return new_edge

    def update_network_from(self, node_id: str, prev_node_id: str):
        """
        This method updates the dimensions of the network starting from
        the given node, giving to it, as input dimensions, the output
        dimensions of the given previous node.
       :param node_id: str
       :param prev_node_id: str
        """
        node = self.project.NN.nodes[node_id]
        in_dim = self.project.NN.nodes[prev_node_id].out_dim

        if node is not None:
            while node is not None:
                NodeOps.update_node_input(node, in_dim)

                self.disconnected_network[node.identifier].block_data = NodeOps.node2data(node)[0]
                self.disconnected_network[node.identifier].in_dim = in_dim
                self.disconnected_network[node.identifier].update_labels()

                in_dim = node.out_dim
                node = self.project.NN.get_next_node(node)

    def insert_node(self, prev_node_id: str, middle_node_id: str) -> bool:
        """
        This method inserts the node given as middle node right after the other
        node given as parameter.
        :param prev_node_id: str
        :param middle_node_id: str
        :return: bool

        """

        # Return false if the two nodes are the same, of if one of them is not
        # in the network
        if prev_node_id == middle_node_id:
            return False
        elif middle_node_id not in self.project.NN.nodes.keys():
            return False
        elif prev_node_id not in self.project.NN.nodes.keys():
            return False
        else:
            try:
                prev_node = self.project.NN.nodes[prev_node_id]
                middle_node_block = self.disconnected_network[middle_node_id]
                # Creating the node to insert with the input dimensions taken from
                # the previous node
                middle_node = NodeOps.create_node(middle_node_block.node.class_name,
                                                  middle_node_block.block_id,
                                                  middle_node_block.block_data,
                                                  in_dim=prev_node.out_dim)

                next_node = self.project.NN.get_next_node(prev_node)

                # Adjusting the network
                self.project.NN.edges[prev_node_id] = []
                self.project.NN.edges[prev_node_id].append(middle_node.identifier)
                self.project.NN.edges[middle_node_id] = []
                self.project.NN.edges[middle_node_id].append(next_node.identifier)
                self.project.NN.nodes[middle_node_id] = middle_node

                # Updating the network
                self.update_network_from(next_node.identifier,
                                         middle_node.identifier)
                return True

            except Exception as e:
                raise Exception(e)

    def render(self, nn: NeuralNetwork) -> tuple:
        """
        This method constructs the network to draw.
       :param nn: NeuralNetwork
        """
        self.disconnected_network = {}
        edges = list()
        for node in nn.nodes.values():
            for block_type in self.blocks.values():
                if "<class 'nodes." + block_type.class_name + "'>" \
                        == str(type(node)):
                    # Creating the graphic block
                    block = Block(node.identifier, block_type)
                    block_data = NodeOps.node2data(node)
                    block.block_data = block_data[0]
                    block.in_dim = block_data[1]

                    self.disconnected_network[node.identifier] = block

        # Fetching connections
        for node, destination in nn.edges.items():
            if len(destination) > 0:
                edges.append((node, destination[0]))

        return self.disconnected_network, edges
