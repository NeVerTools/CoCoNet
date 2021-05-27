import math

import numpy as np
import torch

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
            The name of the node type.
        node_id: str
            The unique identifier of the node.
        data: dict
            The dictionary containing the node data.
        in_dim: tuple
            The input of the node.

        Returns
        ----------
        LayerNode
            The concrete LayerNode object.

        """

        if class_name == 'ReLUNode':
            node = ReLUNode(node_id,
                            in_dim)
        elif class_name == 'FullyConnectedNode':
            node = FullyConnectedNode(node_id,
                                      in_dim,
                                      in_dim[-1],
                                      data["out_features"],
                                      Tensor((data["out_features"], in_dim[-1])),
                                      Tensor((data["out_features"],)))
        elif class_name == 'BatchNormNode':
            node = BatchNormNode(node_id,
                                 in_dim, data["num_features"],
                                 Tensor((data["num_features"],)),
                                 Tensor((data["num_features"],)),
                                 data["running_mean"],
                                 data["running_var"], data["eps"],
                                 data["momentum"], data["affine"],
                                 data["track_running_stats"])
        elif class_name == 'AveragePoolNode':
            node = AveragePoolNode(node_id,
                                   in_dim,
                                   data["kernel_size"],
                                   data["stride"],
                                   data["padding"],
                                   data["ceil_mode"],
                                   data["count_include_pad"])
        elif class_name == 'ConvNode':
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
        elif class_name == 'MaxPoolNode':
            node = MaxPoolNode(node_id,
                               in_dim,
                               data["kernel_size"],
                               data["stride"],
                               data["padding"],
                               data["dilation"],
                               data["ceil_mode"],
                               data["return_indices"])
        elif class_name == 'LRNNode':
            node = LRNNode(node_id,
                           in_dim,
                           data["size"],
                           data["alpha"],
                           data["beta"],
                           data["k"])
        elif class_name == 'SoftMaxNode':
            node = SoftMaxNode(node_id,
                               in_dim,
                               data["axis"])
        elif class_name == 'UnsqueezeNode':
            node = UnsqueezeNode(node_id,
                                 in_dim,
                                 data["axes"])
        elif class_name == 'FlattenNode':
            node = FlattenNode(node_id,
                               in_dim,
                               data["axis"])
        elif class_name == 'DropoutNode':
            node = DropoutNode(node_id,
                               in_dim,
                               data["p"])
        elif class_name == 'ReshapeNode':
            node = ReshapeNode(node_id,
                               in_dim,
                               data["shape"])
        else:
            raise Exception(class_name + " node not implemented")

        return node

    @staticmethod
    def update_node_input(node: LayerNode, in_dim: tuple) -> LayerNode:
        """
        This method updates the in_dim and out_dim parameters of
        the node passed. A number of controls is performed and the
        node is returned.

        Parameters
        ----------
        node: LayerNode
            The node to update.
        in_dim: tuple
            The input shape.

        Returns
        ----------
        LayerNode
            The updated node.

        """

        if type(in_dim) != tuple and len(in_dim) < 0:
            raise Exception(f'{LayerNode.__class__.__name__} - empty input.')
        else:
            out = ()
            if isinstance(node, ReLUNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim
                    node.out_dim = in_dim

            elif isinstance(node, FullyConnectedNode):
                if len(in_dim) > 0 and in_dim[-1] == node.in_features:
                    node.in_dim = in_dim

                    if len(in_dim) > 1:
                        for i in range(len(in_dim)):
                            out += (in_dim[i],)
                    out += (node.out_features,)
                    node.out_dim = out
                else:
                    raise Exception('FullyConnectedNode - Wrong input: last value of'
                                    f'input should be {node.in_features}')

            elif isinstance(node, BatchNormNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim
                    node.out_dim = in_dim

            elif isinstance(node, AveragePoolNode):
                if len(in_dim) > 1:
                    node.in_dim = in_dim

                    if len(in_dim) > 2:
                        for i in range(len(in_dim) - 2):
                            out += (in_dim[i],)

                    stride_size = (float(node.stride[0]), float(node.stride[1]))
                    if node.ceil_mode:
                        for i in range(2):
                            out += (math.ceil((in_dim[-2 + i] + 2 * node.padding[i] - node.kernel_size[i]) /
                                              stride_size[i]) + 1,)
                    else:
                        for i in range(2):
                            out += (math.floor((in_dim[-2 + i] + 2 * node.padding[i] - node.kernel_size[i]) /
                                               stride_size[i]) + 1,)
                    node.out_dim = out
                else:
                    raise Exception('AveragePoolNode - Wrong input size: should be > 1')

            elif isinstance(node, ConvNode):
                if len(in_dim) > 2 and 0 < in_dim[-3] == node.in_channels:
                    node.in_dim = in_dim

                    if len(in_dim) > 3:
                        for i in range(len(in_dim) - 3):
                            out += (in_dim[i],)
                    out += (node.out_channels,)

                    stride_size = (float(node.stride[0]), float(node.stride[1]))
                    for i in range(2):
                        out += (math.floor((in_dim[-2 + i] + 2 * node.padding[i] - node.dilation[i] *
                                            (node.kernel_size[i] - 1) - 1) / stride_size[i]) + 1,)
                    node.out_dim = out
                else:
                    raise Exception('ConvNode - Wrong input size: should be > 2'
                                    f'and the third last element should be {node.in_channels}')

            elif isinstance(node, MaxPoolNode):
                if len(in_dim) > 1:
                    node.in_dim = in_dim

                    if len(in_dim) > 2:
                        for i in range(len(in_dim) - 2):
                            out += (in_dim[i],)

                    stride_size = (float(node.stride[0]), float(node.stride[1]))
                    if node.ceil_mode:
                        for i in range(2):
                            out += (math.ceil(((in_dim[-2 + i] + 2 * node.padding[i] - node.dilation[i] * (
                                    node.kernel_size[i] - 1) - 1) / stride_size[i])) + 1,)
                    else:
                        for i in range(2):
                            out += (math.floor(((in_dim[-2 + i] + 2 * node.padding[i] - node.dilation[i] * (
                                    node.kernel_size[i] - 1) - 1) / stride_size[i])) + 1,)
                    node.out_dim = out
                else:
                    raise Exception('MaxPoolNode - Wrong input size: should be > 1')

            elif isinstance(node, LRNNode):
                if len(in_dim) >= 2:
                    node.in_dim = in_dim
                    node.out_dim = in_dim
                else:
                    raise Exception('LRNNode - Wrong input size: should be > 2')

            elif isinstance(node, SoftMaxNode):
                if len(in_dim) >= 0:
                    node.in_dim = in_dim
                    node.out_dim = in_dim
                else:
                    raise Exception('SoftMaxNode - Wrong input.')

            elif isinstance(node, UnsqueezeNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim

                    if len(node.axes) > 0:
                        out = list(in_dim)
                        ax = list(node.axes)
                        for i in range(len(ax)):
                            if (-len(in_dim) - 1) <= ax[i] <= len(in_dim):
                                if -len(in_dim) <= ax[i] < 0:
                                    ax[i] += len(in_dim) + 1
                                out.insert((ax[i] + 1), 1)
                                node.out_dim = tuple(out)
                else:
                    raise Exception('SoftMaxNode - Wrong input size: empty.')

            elif isinstance(node, FlattenNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim
                    # TODO HOW TO COMPUTE OUT_DIM?
                else:
                    raise Exception('FlattenNode - Wrong input size: empty.')

            elif isinstance(node, DropoutNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim
                    node.out_dim = in_dim
                else:
                    raise Exception('DropoutNode - Wrong input size: empty.')

            elif isinstance(node, ReshapeNode):
                if len(in_dim) > 0:
                    node.in_dim = in_dim
                    try:
                        out_dim = torch.reshape(torch.randn(in_dim), node.shape)
                    except Exception:
                        raise Exception(f'shape [{node.shape}] is invalid for input of dimension [{in_dim}]')
                    for i in range(len(out_dim.shape)):
                        out += (out_dim.shape[i],)
                    node.out_dim = out
                else:
                    raise Exception('ReshapeNode - Wrong input size: empty.')

            return node

    @staticmethod
    def update_node_data(node: LayerNode, data: dict) -> LayerNode:
        """
        This method updates the node data with the ones contained
        in the dictionary. For each parameter a number of controls
        is performed and the node is finally returned.

        Parameters
        ----------
        node: LayerNode
            The node to update.
        data: dict
            The new data to save.

        Returns
        ----------
        LayerNode
            The updated node.

        """

        if isinstance(node, FullyConnectedNode):
            if type(data["in_features"]) == int and data["in_features"] > 0:
                node.in_features = data["in_features"]
            else:
                raise Exception('FullyConnectedNode - Wrong "in_features" value, should be int and > 0')

            if type(data["out_features"]) == int and data["out_features"] > 0:
                node.out_features = data["out_features"]
            else:
                raise Exception('FullyConnectedNode - Wrong "out_features" value, should be int and > 0')

            weight = data["weight"]
            if weight is None:
                weight = np.random.normal(size=[node.out_features, node.in_features])
            if weight.shape == (node.out_features, node.in_features):
                node.weight = weight
            else:
                raise Exception('FullyConnectedNode - Wrong weight dimension.')

            bias = data["bias"]
            if bias is None:
                bias = np.random.normal(size=[node.out_features])
            if bias.shape == (node.out_features,):
                node.bias = bias
            else:
                raise Exception('FullyConnectedNode - Wrong bias dimension.')

        elif isinstance(node, BatchNormNode):
            for dim in range(len(node.in_dim)):
                if node.in_dim[dim] == data["num_features"]:
                    node.num_features = data["num_features"]

            if node.num_features is None:
                raise Exception(f'BatchNormNode - Wrong input, num_features = {node.num_features}'
                                f'not feasible with input shape = {node.in_dim}')

            if data["track_running_stats"] and data["running_mean"] is None and data["running_var"] is None:
                data["running_mean"] = np.ones(node.num_features)
                data["running_var"] = np.zeros(node.num_features)

            weight = data["weight"]
            if weight is None:
                weight = np.ones(node.num_features)
            if weight.shape == (node.num_features,):
                node.weight = weight
            else:
                raise Exception('BatchNormNode - Wrong weight dimension.')

            bias = data["bias"]
            if bias is None:
                bias = np.zeros(node.num_features)
            if bias.shape == (node.num_features,):
                node.bias = bias
            else:
                raise Exception('BatchNormNode - Wrong bias dimension.')

            node.running_mean = data["running_mean"]
            node.running_var = data["running_var"]
            node.track_running_stats = data["track_running_stats"]
            node.eps = data["eps"]
            node.momentum = data["momentum"]
            node.affine = data["affine"]

        elif isinstance(node, AveragePoolNode):
            if data["stride"] is None:
                data["stride"] = data["kernel_size"]

            kernel = data["kernel"]
            if type(kernel) == tuple:
                if len(kernel) == 2 and kernel[0] > 0 and kernel[1] > 0:
                    node.kernel_size = kernel
                elif len(kernel) == 1 and kernel[0] > 0:
                    kernel = (kernel[0], kernel[0])
                    node.kernel_size = kernel
                else:
                    raise Exception('AveragePoolNode - Wrong kernel size type, must be int'
                                    'or tuple with values > 0')
            else:
                raise Exception('AveragePoolNode - Wrong kernel size type, must be int or tuple with values > 0')

            stride = data["stride"]
            if type(stride) == tuple:
                if len(stride) == 2 and stride[0] > 0 and stride[1] > 0:
                    node.stride = stride
                elif len(stride) == 1 and stride[0] > 0:
                    stride = (stride[0], stride[0])
                    node.stride = stride
                else:
                    raise Exception('AveragePoolNode - Wrong stride type, must be int'
                                    'or tuple with values > 0')
            else:
                raise Exception('AveragePoolNode - Wrong stride type, must be int or tuple with values > 0')

            padding = data["padding"]
            if type(padding) == tuple:
                if len(padding) == 1:
                    padding = (padding[0], padding[0])
                half = list(kernel)
                half[0] = half[0] / 2.0
                half[1] = half[1] / 2.0

                if padding[0] <= half[0] and padding[1] <= half[1]:
                    node.padding = padding
                else:
                    raise Exception(f'AveragePoolNode - padding should be smaller than half of kernel size,'
                                    f'but got padW = {padding[1]}, padH = {padding[0]},'
                                    f'kW = {kernel[1]}, kH = {kernel[0]}')
            else:
                raise Exception('AveragePoolNode - padding type must be int or tuple.')

            count_include_pad = data["count_include_pad"]
            if type(count_include_pad) is bool:
                node.count_include_pad = count_include_pad  # Do not modify shape
            else:
                raise Exception('AveragePoolNode - count_include_pad must be bool.')

            ceil_mode = data["ceil_mode"]
            if type(ceil_mode) is bool:
                node.ceil_mode = ceil_mode
            else:
                raise Exception('AveragePoolNode - ceil_mode must be bool.')

        elif isinstance(node, ConvNode):
            in_channels = data["in_channels"]
            if type(in_channels) == int and in_channels > 0:
                node.in_channels = in_channels
            else:
                raise Exception('ConvNode - in_channels must be int and > 0')

            out_channels = data["out_channels"]
            if type(out_channels) == int and out_channels > 0:
                node.out_channels = out_channels
            else:
                raise Exception('ConvNode - Wrong out_channel value, must be int > 0')

            bias = data["bias"]
            if bias is None:
                bias = np.random.normal(size=[out_channels])
            if type(bias) == Tensor and bias.size == out_channels:
                node.bias = bias
            else:
                raise Exception(f'ConvNode - Wrong bias value, it should have shape of {out_channels}')

            weight = data["weight"]
            size = (out_channels, in_channels)
            size += data["kernel_size"]
            if weight is None:
                weight = np.random.normal(size=size)
            if type(weight) == Tensor and weight.shape == size:
                node.weight = weight
            else:
                raise Exception(f'ConvNode - Wrong weight value, it should have shape = {size}')

            kernel = data["kernel_size"]
            if type(kernel) == tuple:
                if len(kernel) == 2 and kernel[0] > 0 and kernel[1] > 0:
                    node.kernel_size = kernel
                elif len(kernel) == 1 and kernel[0] > 0:
                    kernel = (kernel[0], kernel[0])
                    node.kernel_size = kernel
                else:
                    raise Exception('ConvNode - Wrong kernel size value, must be > 0')
            else:
                raise Exception('ConvNode - Wrong kernel size value, must be > 0')

            stride = data["stride"]
            if type(stride) == tuple:
                if len(stride) == 2 and stride[0] > 0 and stride[1] > 0:
                    node.stride = stride
                elif len(stride) == 1 and stride[0] > 0:
                    stride = (stride[0], stride[0])
                    node.stride = stride
                else:
                    raise Exception('ConvNode - Wrong stride value, must be > 0')
            else:
                raise Exception('ConvNode - Wrong stride value, must be > 0')

            padding = data["padding"]
            if type(padding) == tuple:
                if len(padding) == 2 and padding[0] >= 0 and padding[1] >= 0:
                    node.padding = padding
                elif len(padding) == 1 and padding[0] >= 0:
                    padding = (padding[0], padding[0])
                    node.padding = padding
                else:
                    raise Exception('ConvNode - Wrong padding value, must be >= 0')
            else:
                raise Exception('ConvNode - Wrong padding value, must be >= 0')

            dilation = data["dilation"]
            if type(dilation) == tuple:
                if len(dilation) == 2 and dilation[0] >= 0 and dilation[1] >= 0:
                    node.dilation = dilation
                elif len(dilation) == 1 and dilation[0] >= 0:
                    dilation = (dilation[0], dilation[0])
                    node.dilation = dilation
                else:
                    raise Exception('ConvNode - Wrong dilation value, must be >= 0')
            else:
                raise Exception('ConvNode - Wrong dilation value, must be >= 0')

            groups = data["groups"]
            if groups > 0 and in_channels % groups == 0 and out_channels % groups == 0:
                node.groups = groups
            else:
                raise Exception('ConvNode - Wrong groups value,'
                                'in_channels and out_channels must be divisible by groups')

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
