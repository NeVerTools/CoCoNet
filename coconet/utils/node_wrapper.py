from pynever.nodes import LayerNode, ReLUNode, SigmoidNode, FullyConnectedNode, BatchNormNode, AveragePoolNode, \
    ConvNode, MaxPoolNode, LRNNode, SoftMaxNode, UnsqueezeNode, FlattenNode, DropoutNode, ReshapeNode, ELUNode, \
    CELUNode, LeakyReLUNode, TanhNode


class NodeFactory:
    @staticmethod
    def create_node(class_name: str, node_id: str, data: dict, in_dim: tuple) -> LayerNode:
        """
        This method builds the corresponding LayerNode given the class name,
        a unique identifier and the node data. If the class name is not implemented,
        raises an Exception

        Parameters
        ----------
        class_name : str
            The name of the node type.
        node_id : str
            The unique identifier of the node.
        data : dict
            The dictionary containing the node data.
        in_dim : tuple
            The input of the node.

        Returns
        ----------
        LayerNode
            The concrete LayerNode object.

        """

        if class_name == 'ReLUNode':
            node = ReLUNode(node_id,
                            in_dim)
        elif class_name == 'ELUNode':
            node = ELUNode(node_id,
                           in_dim,
                           alpha=data['alpha'])
        elif class_name == 'CELUNode':
            node = CELUNode(node_id,
                            in_dim,
                            alpha=data['alpha'])
        elif class_name == 'LeakyReLUNodeNode':
            node = LeakyReLUNode(node_id,
                                 in_dim,
                                 negative_slope=data['negative_slope'])
        elif class_name == 'SigmoidNode':
            node = SigmoidNode(node_id,
                               in_dim)
        elif class_name == 'TanhNode':
            node = TanhNode(node_id,
                            in_dim)
        elif class_name == 'FullyConnectedNode':
            node = FullyConnectedNode(node_id,
                                      in_dim,
                                      data['out_features'],
                                      weight=None, bias=None,
                                      has_bias=data['has_bias'])
        elif class_name == 'BatchNormNode':
            node = BatchNormNode(node_id,
                                 in_dim, weight=None,
                                 bias=None,
                                 running_mean=None,
                                 running_var=None, eps=data['eps'],
                                 momentum=data['momentum'], affine=data['affine'],
                                 track_running_stats=data['track_running_stats'])
        elif class_name == 'AveragePoolNode':
            node = AveragePoolNode(node_id,
                                   in_dim,
                                   data['kernel_size'],
                                   data['stride'],
                                   data['padding'],
                                   data['ceil_mode'],
                                   data['count_include_pad'])
        elif class_name == 'ConvNode':
            node = ConvNode(node_id,
                            in_dim,
                            data['out_channels'],
                            data['kernel_size'],
                            data['stride'],
                            data['padding'],
                            data['dilation'],
                            data['groups'],
                            data['has_bias'],
                            bias=None, weight=None)
        elif class_name == 'MaxPoolNode':
            node = MaxPoolNode(node_id,
                               in_dim,
                               data['kernel_size'],
                               data['stride'],
                               data['padding'],
                               data['dilation'],
                               data['ceil_mode'],
                               data['return_indices'])
        elif class_name == 'LRNNode':
            node = LRNNode(node_id,
                           in_dim,
                           data['size'],
                           data['alpha'],
                           data['beta'],
                           data['k'])
        elif class_name == 'SoftMaxNode':
            node = SoftMaxNode(node_id,
                               in_dim,
                               data['axis'])
        elif class_name == 'UnsqueezeNode':
            node = UnsqueezeNode(node_id,
                                 in_dim,
                                 data['axes'])
        elif class_name == 'FlattenNode':
            node = FlattenNode(node_id,
                               in_dim,
                               data['axis'])
        elif class_name == 'DropoutNode':
            node = DropoutNode(node_id,
                               in_dim,
                               data['p'])
        elif class_name == 'ReshapeNode':
            node = ReshapeNode(node_id,
                               in_dim,
                               data['shape'])
        else:
            raise Exception(class_name + ' node not implemented')

        return node
