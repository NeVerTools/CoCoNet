from typing import Union

from coconet.core.controller.pynevertemp.nodes import LayerNode, ReLUNode, FullyConnectedNode, BatchNormNode, \
    AveragePoolNode, ConvNode, MaxPoolNode, LRNNode, SoftMaxNode, UnsqueezeNode, FlattenNode, DropoutNode, ReshapeNode
from coconet.core.controller.pynevertemp.tensor import Tensor


class NodeOps:
    @staticmethod
    def create_node(class_name: str, node_id: str, data: dict, in_dim: tuple) -> LayerNode:
        """
        This method builds the corresponding LayerNode given the
        class name, the unique identifier and the node data.
        If the class name is not part of the implemented set, an
        exception is raised.

        Parameters
        ----------
        class_name: str
        node_id: str
        data: dict
        in_dim: tuple

        Returns
        ----------
        LayerNode

        """

        if class_name == "ReLUNode":
            node = ReLUNode(node_id,
                            in_dim)
        elif class_name == "FullyConnectedNode":
            node = FullyConnectedNode(node_id,
                                      in_dim,
                                      in_dim[-1],
                                      data["out_features"],
                                      Tensor((data["out_features"], in_dim[-1])),
                                      Tensor((data["out_features"],)))
        elif class_name == "BatchNormNode":
            node = BatchNormNode(node_id,
                                 in_dim, data["num_features"],
                                 Tensor((data["num_features"],)),
                                 Tensor((data["num_features"],)),
                                 data["running_mean"],
                                 data["running_var"], data["eps"],
                                 data["momentum"], data["affine"],
                                 data["track_running_stats"])
        elif class_name == "AveragePoolNode":
            node = AveragePoolNode(node_id,
                                   in_dim,
                                   data["kernel_size"],
                                   data["stride"],
                                   data["padding"],
                                   data["ceil_mode"],
                                   data["count_include_pad"])
        elif class_name == "ConvNode":
            node = ConvNode(node_id,
                            in_dim,
                            data["in_channels"],
                            data["out_channels"],
                            data["kernel_size"],
                            data["stride"],
                            data["padding"],
                            data["dilation"],
                            data["groups"],
                            data["has_bias"],
                            data["bias"],
                            data["weight"])
        elif class_name == "MaxPoolNode":
            node = MaxPoolNode(node_id,
                               in_dim,
                               data["kernel_size"],
                               data["stride"],
                               data["padding"],
                               data["dilation"],
                               data["ceil_mode"],
                               data["return_indices"])
        elif class_name == "LRNNode":
            node = LRNNode(node_id,
                           in_dim,
                           data["size"],
                           data["alpha"],
                           data["beta"],
                           data["k"])
        elif class_name == "SoftMaxNode":
            node = SoftMaxNode(node_id,
                               in_dim,
                               data["axis"])
        elif class_name == "UnsqueezeNode":
            node = UnsqueezeNode(node_id,
                                 in_dim,
                                 data["axes"])
        elif class_name == "FlattenNode":
            node = FlattenNode(node_id,
                               in_dim,
                               data["axis"])
        elif class_name == "DropoutNode":
            node = DropoutNode(node_id,
                               in_dim,
                               data["p"])
        elif class_name == "ReshapeNode":
            node = ReshapeNode(node_id,
                               in_dim,
                               data["shape"])
        else:
            raise Exception(class_name + " node not implemented")

        return node

    @staticmethod
    def update_node_input(node: LayerNode, in_dim: Union[tuple, int]) -> LayerNode:
        """
        This method updates the node passed as its input changes.

        Parameters
        ----------
        node: LayerNode
        in_dim: Union[tuple, int]

        Returns
        ----------
        LayerNode

        """

        node.in_dim = in_dim
        return node

    @staticmethod
    def update_node_data(node: LayerNode, data: dict) -> LayerNode:
        """
        This method updates the given node with the new data given.
        The updated node is returned.

        Parameters
        ----------
        node: LayerNode
        data: dict

        Returns
        ----------
        LayerNode

        """

        if isinstance(node, FullyConnectedNode):
            node.update(node.in_dim[-1],
                        data["out_features"],
                        node.in_dim,
                        Tensor((data["out_features"], node.in_dim[-1])),
                        Tensor((data["out_features"],)))
        elif isinstance(node, BatchNormNode):
            node.update(data["num_features"],
                        node.in_dim,
                        Tensor((data["num_features"],)),
                        Tensor((data["num_features"],)),
                        data["running_mean"],
                        data["running_var"],
                        data["eps"],
                        data["momentum"],
                        data["affine"],
                        data["track_running_stats"])
        elif isinstance(node, AveragePoolNode):
            node.update(node.in_dim,
                        data["kernel_size"],
                        data["ceil_mode"],
                        data["padding"],
                        data["stride_size"],
                        data["count_include_pad"])
        elif isinstance(node, ConvNode):
            node.update(node.in_dim,
                        data["in_channels"],
                        data["out_channels"],
                        data["kernel_size"],
                        data["dilation"],
                        data["groups"],
                        Tensor((data["num_features"],)),
                        data["padding"],
                        data["stride_size"])
        elif isinstance(node, MaxPoolNode):
            node.update(node.in_dim,
                        data["kernel_size"],
                        data["dilation"],
                        data["stride_size"],
                        data["padding"],
                        data["return_indices"],
                        data["ceil_mode"])
        elif isinstance(node, LRNNode):
            node.update(node.in_dim,
                        data["size"],
                        data["alpha"],
                        data["beta"],
                        Tensor((data["size"],)))
        elif isinstance(node, SoftMaxNode):
            node.update(data["dim"],
                        node.in_dim)
        elif isinstance(node, UnsqueezeNode):
            node.update(data["axis"],
                        node.in_dim)
        elif isinstance(node, FlattenNode):
            node.update(node.in_dim,
                        data["start_dim"],
                        data["end_dim"])
        elif isinstance(node, DropoutNode):
            node.update(node.in_dim,
                        data["p"],
                        data["in_place"])
        elif isinstance(node, ReshapeNode):
            node.update(node.in_dim,
                        data["shape"])

        return node

    @staticmethod
    def node2data(node: LayerNode) -> tuple:
        """
        This method creates a tuple of dictionaries with the information
        to draw a new block.

        Parameters
        ----------
        node: LayerNode

        Returns
        ----------
        tuple

        """

        data = dict()

        if isinstance(node, FullyConnectedNode):
            data["in_features"] = node.in_features
            data["out_features"] = node.out_features
            data["weight"] = node.weight
            data["bias"] = node.bias
            data["has_bias"] = node.has_bias
        elif isinstance(node, BatchNormNode):
            data["num_features"] = node.num_features
            data["weight"] = node.weight
            data["bias"] = node.bias
            data["running_mean"] = node.running_mean
            data["running_var"] = node.running_var
            data["eps"] = node.eps
            data["momentum"] = node.momentum
            data["affine"] = node.affine
            data["track_running_stats"] = node.track_running_stats
        elif isinstance(node, AveragePoolNode):
            data["kernel_size"] = node.kernel_size
            data["stride"] = node.stride
            data["padding"] = node.padding
            data["ceil_mode"] = node.ceil_mode
            data["count_include_pad"] = node.count_include_pad
        elif isinstance(node, ConvNode):
            data["in_channels"] = node.in_channels
            data["out_channels"] = node.out_channels
            data["kernel_size"] = node.kernel_size
            data["stride"] = node.stride
            data["padding"] = node.padding
            data["dilation"] = node.dilation
            data["groups"] = node.groups
            data["has_bias"] = node.has_bias
            data["bias"] = node.bias
            data["weight"] = node.weight
        elif isinstance(node, MaxPoolNode):
            data["kernel_size"] = node.kernel_size
            data["stride"] = node.stride
            data["padding"] = node.padding
            data["dilation"] = node.dilation
            data["ceil_mode"] = node.ceil_mode
            data["return_indices"] = node.return_indices
        elif isinstance(node, LRNNode):
            data["size"] = node.size
            data["alpha"] = node.alpha
            data["beta"] = node.beta
            data["k"] = node.k
        elif isinstance(node, SoftMaxNode):
            data["axis"] = node.axis
        elif isinstance(node, UnsqueezeNode):
            data["axes"] = node.axes
        elif isinstance(node, FlattenNode):
            data["axis"] = node.axis
        elif isinstance(node, DropoutNode):
            data["p"] = node.p
        elif isinstance(node, ReshapeNode):
            data["shape"] = node.shape
            data["allow_zero"] = node.allow_zero

        return data, node.in_dim
