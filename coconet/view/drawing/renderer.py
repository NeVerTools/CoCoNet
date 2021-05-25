import copy
from typing import Union

from PyQt5.QtCore import pyqtSignal

from coconet.core.controller.project import Project
from coconet.core.controller.pynevertemp.networks import NeuralNetwork
from coconet.core.controller.pynevertemp.nodes import ReLUNode, FullyConnectedNode, BatchNormNode, AveragePoolNode, \
    ConvNode, MaxPoolNode, LRNNode, SoftMaxNode, UnsqueezeNode, FlattenNode, DropoutNode, ReshapeNode, LayerNode
from coconet.core.controller.pynevertemp.tensor import Tensor
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
    create_node(str, str, dict, tuple, int, int):
        explained below
    update_node_input(str, LayerNode, tuple):
        explained below
    update_node_data(str, LayerNode, dict):
        explained below
    draw_network(NeuralNetwork):
        explained below
    has_nodes_before(str):
        explained below
    layer_node_to_data(LayerNode):
        explained below

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
                node = self.create_node(block.node.class_name, node_id,
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
            st_node = self.create_node(origin_block.node.class_name,
                                       origin_block.block_id,
                                       origin_block.block_data,
                                       origin_block.in_dim)
            nd_node = self.create_node(destination_block.node.class_name,
                                       destination_block.block_id,
                                       destination_block.block_data,
                                       st_node.out_dim)
            self.project.NN.add_node(st_node)
            self.project.NN.add_node(nd_node)

            self.disconnected_network[nd_node.identifier].block_data = self.layer_node_to_data(nd_node)[0]
            self.disconnected_network[nd_node.identifier].in_dim = nd_node.in_dim
            self.disconnected_network[nd_node.identifier].update_labels()

            return True
        elif origin_id not in self.project.NN.nodes and \
                destination_id not in self.project.NN.nodes:
            # If the network is not empty but the connection to draw is
            # between two new nodes, the connection is drawn,
            # but probably the network will not be saved
            st_node = self.create_node(origin_block.node.class_name,
                                       origin_block.block_id,
                                       origin_block.block_data,
                                       origin_block.in_dim)
            nd_node = self.create_node(destination_block.node.class_name,
                                       destination_block.block_id,
                                       destination_block.block_data,
                                       st_node.out_dim)

            self.project.NN.nodes[origin_id] = st_node
            self.project.NN.nodes[destination_id] = nd_node
            self.project.NN.edges[origin_id] = [destination_id]
            self.project.NN.edges[destination_id] = []

            self.disconnected_network[nd_node.identifier].block_data = self.layer_node_to_data(nd_node)[0]
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
                orig_node = self.create_node(origin_block.node.class_name,
                                             origin_block.block_id,
                                             origin_block.block_data,
                                             origin_block.in_dim)
                self.project.NN.nodes[origin_id] = orig_node
                self.project.NN.edges[origin_id] = []
            else:
                orig_node = self.project.NN.nodes[origin_id]

            if destination_id not in self.project.NN.nodes:
                dest_node = self.create_node(destination_block.node.class_name,
                                             destination_block.block_id,
                                             destination_block.block_data,
                                             orig_node.out_dim)
                self.project.NN.nodes[destination_id] = dest_node
                self.project.NN.edges[destination_id] = []
            else:
                dest_node = self.project.NN.nodes[destination_id]
                self.update_node_input(dest_node, orig_node.out_dim)
                # If the destination had other blocks after it, they are updated
                if len(self.project.NN.edges[destination_id]) > 0:
                    self.update_network_from(destination_id, origin_id)

            self.project.NN.edges[origin_id].append(destination_id)

            self.disconnected_network[dest_node.identifier].block_data = self.layer_node_to_data(dest_node)[0]
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
                    self.update_node_input(node, in_dim)
                self.update_node_data(node, data)

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
            self.update_node_data(node, old_data)
            if in_dim is not None:
                self.update_node_input(node, old_in_dim)

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
                self.update_node_input(node, in_dim)

                self.disconnected_network[node.identifier].block_data = self.layer_node_to_data(node)[0]
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
                middle_node = self.create_node(middle_node_block.node.class_name,
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

    @staticmethod
    def create_node(class_name: str, node_id: str, data: dict, in_dim) -> LayerNode:
        """
        This method creates a LayerNode object of the class given by parameter,
        eventually with other properties to initialize it.

        :param class_name: str
        :param node_id: str
        :param data: dict
        :param in_dim: tuple
        :return: LayerNode

        """

        if class_name == "ReLUNode":
            node = ReLUNode(node_id, in_dim)
        elif class_name == "FullyConnectedNode":
            node = FullyConnectedNode(node_id, in_dim, in_dim[-1], data["out_features"],
                                      Tensor((data["out_features"], in_dim[-1])),
                                      Tensor((data["out_features"],)))
        elif class_name == "BatchNormNode":
            node = BatchNormNode(node_id, in_dim, data["num_features"],
                                 Tensor((data["num_features"],)),
                                 Tensor((data["num_features"],)),
                                 data["running_mean"],
                                 data["running_var"], data["eps"],
                                 data["momentum"], data["affine"],
                                 data["track_running_stats"])
        elif class_name == "AveragePoolNode":
            node = AveragePoolNode(node_id, in_dim, data["kernel_size"],
                                   data["stride"], data["padding"],
                                   data["ceil_mode"], data["count_include_pad"])
        elif class_name == "ConvNode":
            node = ConvNode(node_id, in_dim, data["in_channels"], data["out_channels"],
                            data["kernel_size"], data["stride"], data["padding"],
                            data["dilation"], data["groups"], data["has_bias"],
                            data["bias"], data["weight"])
        elif class_name == "MaxPoolNode":
            node = MaxPoolNode(node_id, in_dim, data["kernel_size"],
                               data["stride"], data["padding"],
                               data["dilation"], data["ceil_mode"],
                               data["return_indices"])
        elif class_name == "LRNNode":
            node = LRNNode(node_id, in_dim, data["size"], data["alpha"],
                           data["beta"], data["k"])
        elif class_name == "SoftMaxNode":
            node = SoftMaxNode(node_id, in_dim, data["axis"])
        elif class_name == "UnsqueezeNode":
            node = UnsqueezeNode(node_id, in_dim, data["axes"])
        elif class_name == "FlattenNode":
            node = FlattenNode(node_id, in_dim, data["axis"])
        elif class_name == "DropoutNode":
            node = DropoutNode(node_id, in_dim, data["p"])
        elif class_name == "ReshapeNode":
            node = ReshapeNode(node_id, in_dim, data["shape"])
        else:
            raise Exception(class_name + " node not implemented")

        return node

    @staticmethod
    def update_node_data(node: LayerNode, data: dict):
        """
        This class updates the given node when the user edits its properties.

        :param node: LayerNode
        :param data: dict

        """

        if isinstance(node, FullyConnectedNode):
            node.update(node.in_dim[-1], data["out_features"], node.in_dim,
                        Tensor((data["out_features"], node.in_dim[-1])),
                        Tensor((data["out_features"],)))
        elif isinstance(node, BatchNormNode):
            node.update(data["num_features"], node.in_dim,
                        Tensor((data["num_features"],)),
                        Tensor((data["num_features"],)),
                        data["running_mean"],
                        data["running_var"], data["epsilon"],
                        data["momentum"], data["affine"],
                        data["track_running_stats"])
        elif isinstance(node, AveragePoolNode):
            node.update(node.in_dim, data["kernel_size"], data["ceil_mode"],
                        data["padding"], data["stride_size"],
                        data["count_include_pad"])
        elif isinstance(node, ConvNode):
            node.update(node.in_dim, data["in_channels"], data["out_channels"],
                        data["kernel_size"], data["dilation"],
                        data["groups"], Tensor((data["num_features"],)),
                        data["padding"],
                        data["stride_size"])
        elif isinstance(node, MaxPoolNode):
            node.update(node.in_dim, data["kernel_size"], data["dilation"],
                        data["stride_size"], data["padding"],
                        data["return_indices"], data["ceil_mode"])
        elif isinstance(node, LRNNode):
            node.update(node.in_dim, data["size"], data["alpha"], data["beta"],
                        Tensor((data["size"],)))
        elif isinstance(node, SoftMaxNode):
            node.update(data["dim"], node.in_dim)
        elif isinstance(node, UnsqueezeNode):
            node.update(data["axis"], node.in_dim)
        elif isinstance(node, FlattenNode):
            node.update(node.in_dim, data["start_dim"], data["end_dim"])
        elif isinstance(node, DropoutNode):
            node.update(node.in_dim, data["p"], data["in_place"])
        elif isinstance(node, ReshapeNode):
            node.update(node.in_dim, data["shape"])

    @staticmethod
    def update_node_input(node: LayerNode, in_dim: Union[tuple, int]):
        """
        This method updates the node passed as its input changes.

       :param node: LayerNode
       :param in_dim: Union[tuple, int]
        """
        if isinstance(node, FullyConnectedNode):
            node.update(in_dim[-1], node.out_features, in_dim,
                        Tensor((node.out_features, in_dim[-1])), node.bias)
        elif isinstance(node, BatchNormNode):
            node.update(node.num_features, in_dim,
                        node.weight, node.bias, node.running_mean,
                        node.running_var, node.epsilon,
                        node.momentum, node.affine,
                        node.track_running_stats)
        elif isinstance(node, AveragePoolNode):
            node.update(in_dim, node.kernel_size, node.ceil_mode,
                        node.padding, node.stride_size,
                        node.count_include_pad)
        elif isinstance(node, ConvNode):
            node.update(in_dim, node.in_channels, node.out_channels,
                        node.kernel_size, node.dilation,
                        node.groups, node.bias, node.padding,
                        node.stride_size)
        elif isinstance(node, MaxPoolNode):
            node.update(in_dim, node.kernel_size, node.dilation,
                        node.stride_size, node.padding,
                        node.return_indices, node.ceil_mode)
        elif isinstance(node, LRNNode):
            node.update(in_dim, node.size, node.alpha, node.beta,
                        node.bias)
        elif isinstance(node, SoftMaxNode):
            node.update(node.dim, in_dim)
        elif isinstance(node, UnsqueezeNode):
            node.update(node.axis, in_dim)
        elif isinstance(node, FlattenNode):
            node.update(in_dim, node.start_dim, node.end_dim)
        elif isinstance(node, DropoutNode):
            node.update(in_dim, node.p, node.in_place)
        elif isinstance(node, ReshapeNode):
            node.update(in_dim, node.shape)
        elif isinstance(node, ReLUNode):
            node.update(in_dim)

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
                    block_data = self.layer_node_to_data(node)
                    block.block_data = block_data[0]
                    block.in_dim = block_data[1]

                    self.disconnected_network[node.identifier] = block

        # Fetching connections
        for node, destination in nn.edges.items():
            if len(destination) > 0:
                edges.append((node, destination[0]))

        return self.disconnected_network, edges

    @staticmethod
    def layer_node_to_data(node: LayerNode) -> tuple:
        """
        This method creates a tuple of dictionaries with the information
        to draw a new block.
       :param node: LayerNode
       :return: tuple
        """
        data = dict()

        if isinstance(node, FullyConnectedNode):
            data["in_features"] = node.in_features
            data["out_features"] = node.out_features
            data["weight"] = node.weight
            data["bias"] = node.bias
        elif isinstance(node, BatchNormNode):
            data["num_features"] = node.num_features
            data["weight"] = node.weight
            data["bias"] = node.bias
            data["running_mean"] = node.running_mean
            data["running_var"] = node.running_var
            data["epsilon"] = node.eps
            data["momentum"] = node.momentum
            data["affine"] = node.affine
            data["track_running_stats"] = node.track_running_stats
        elif isinstance(node, AveragePoolNode):
            data["stride_size"] = node.stride_size
            data["kernel_size"] = node.kernel_size
            data["ceil_mode"] = node.ceil_mode
            data["padding"] = node.padding
        elif isinstance(node, ConvNode):
            data["in_channels"] = node.in_channels
            data["out_channels"] = node.out_channels
            data["kernel_size"] = node.kernel_size
            data["stride_size"] = node.stride_size
            data["padding"] = node.padding
            data["dilation"] = node.dilation
            data["groups"] = node.groups
            data["bias"] = node.bias
            data["weight"] = node.weight
        elif isinstance(node, MaxPoolNode):
            data["ceil_mode"] = node.ceil_mode
            data["kernel_size"] = node.kernel_size
            data["stride_size"] = node.stride_size
            data["padding"] = node.padding
            data["return_indices"] = node.return_indices
            data["dilation"] = node.dilation
        elif isinstance(node, LRNNode):
            data["alpha"] = node.alpha
            data["beta"] = node.beta
            data["size"] = node.size
        elif isinstance(node, SoftMaxNode):
            data["dim"] = node.dim
        elif isinstance(node, UnsqueezeNode):
            data["axis"] = node.axis
        elif isinstance(node, FlattenNode):
            data["start_dim"] = node.start_dim
            data["end_dim"] = node.end_dim
        elif isinstance(node, DropoutNode):
            data["p"] = node.p
            data["in_place"] = node.in_place
        elif isinstance(node, ReshapeNode):
            data["shape"] = node.shape

        return data, node.in_dim
