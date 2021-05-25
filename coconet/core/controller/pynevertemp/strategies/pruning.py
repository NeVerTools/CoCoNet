import abc
import coconet.core.controller.pynevertemp.networks as networks
import coconet.core.controller.pynevertemp.datasets as datasets
import coconet.core.controller.pynevertemp.strategies.conversion as cv
import coconet.core.controller.pynevertemp.strategies.training as training
import torch
import math
import torch.nn as nn


class PruningStrategy(abc.ABC):
    """
    An abstract class used to represent a Pruning Strategy.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using a pruning strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using a pruning strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset: Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """
        pass


class WeightPruning(PruningStrategy):
    """
    A concrete class used to represent the weight pruning strategy.
    This kind of pruning select the least important weights of the neural network
    of interest and set them to 0.
    We refer to https://arxiv.org/abs/1506.02626 for theoretical details on the strategy.

    Attributes
    ----------
    sparsity_rate : float
        It determines the percentage of neurons which will be removed. It must be a Real number between 0 and 1.
    training_strategy : AdamTraining
        The training strategy to use for pre-training and/or fine-tuning.
    pre_training : bool
        Flag to indicate if the network need to be pre-trained.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using the pruning strategy Weight Pruning.

    """

    def __init__(self, sparsity_rate: float, training_strategy: training.AdamTraining = None,
                 pre_training: bool = False):

        self.sparsity_rate = sparsity_rate
        self.training_strategy = training_strategy
        self.pre_training = pre_training

    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using the pruning strategy Weight Pruning.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset : Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """

        if self.training_strategy is not None and self.pre_training:
            fine_tuning = self.training_strategy.fine_tuning
            self.training_strategy.fine_tuning = False
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.fine_tuning = fine_tuning

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__pruning(py_net)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        if self.training_strategy is not None and self.training_strategy.fine_tuning:
            old_l1_decay = self.training_strategy.l1_decay
            old_batchnorm_decay = self.training_strategy.batchnorm_decay
            self.training_strategy.l1_decay = 0
            self.training_strategy.batchnorm_decay = 0
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.l1_decay = old_l1_decay
            self.training_strategy.batchnorm_decay = old_batchnorm_decay

        return network

    def __pruning(self, net: cv.PyTorchNetwork):
        """
        Procedure for the pruning of the weights of the PyTorchNetwork passed as an argument.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to prune.

        Returns
        ----------
        PyTorchNetwork
            The pruned PyTorchNetwork.

        """

        # We transfer the internal pytorch model to the CPU for the pruning procedure.
        net.pytorch_network.cpu()

        # We compute the number of weights in the network
        num_weights = 0
        for m in net.pytorch_network.modules():

            if isinstance(m, torch.nn.Linear):
                num_weights += m.weight.numel()

        # We copy all the absolute values of the weights in a new tensor and we sort in ascending order
        weights = torch.zeros(num_weights)
        index = 0
        for m in net.pytorch_network.modules():

            if isinstance(m, torch.nn.Linear):
                size = m.weight.numel()
                weights[index:(index + size)] = m.weight.view(-1).abs().clone()
                index += size

        ordered_weights, ordered_indexes = torch.sort(weights)

        # We determine the number of weights we need to set to 0, given the sparsity rate.
        threshold_index = math.floor(num_weights * self.sparsity_rate)

        # We select the weight absolute value we will use as threshold value given the threshold index.
        threshold_value = ordered_weights[threshold_index]

        # We set all the weights of the different layers to 0 if they are less or equal than the threshold value
        # (in absolute value)

        for m in net.pytorch_network.modules():

            if isinstance(m, torch.nn.Linear):
                # The values of the mask are 0 when the corresponding weight is less then the threshold_value (in
                # absolute value), otherwise they are 1
                mask = m.weight.abs().gt(threshold_value).float()
                m.weight.data = torch.mul(m.weight, mask)

        return net


class NetworkSlimming(PruningStrategy):
    """
    A concrete class used to represent the network slimming pruning strategy.
    This kind of pruning select the least important neurons of the neural network
    of interest and eliminates them. It needs a batch normalization layer following each layer
    which should be pruned. We assume that the activation function is always applied after the batch
    normalization layer.
    We refer to https://arxiv.org/abs/1708.06519 for theoretical details on the strategy.

    Attributes
    ----------
    sparsity_rate : float
        It determines the percentage of neurons which will be removed. It must be a Real number between 0 and 1.
    training_strategy : AdamTraining
        The training strategy to use for pre-training and/or fine-tuning.
    pre_training : bool
        Flag to indicate if the network need to be pre-trained.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using the pruning strategy Network Slimming.

    """

    def __init__(self, sparsity_rate: float, training_strategy: training.AdamTraining = None,
                 pre_training: bool = False):

        self.sparsity_rate = sparsity_rate
        self.training_strategy = training_strategy
        self.pre_training = pre_training

    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using the pruning strategy Network Slimming.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset: Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """

        if self.training_strategy is not None and self.pre_training:
            fine_tuning = self.training_strategy.fine_tuning
            self.training_strategy.fine_tuning = False
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.fine_tuning = fine_tuning

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__pruning(py_net)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        if self.training_strategy is not None and self.training_strategy.fine_tuning:
            old_l1_decay = self.training_strategy.l1_decay
            old_batchnorm_decay = self.training_strategy.batchnorm_decay
            self.training_strategy.l1_decay = 0
            self.training_strategy.batchnorm_decay = 0
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.l1_decay = old_l1_decay
            self.training_strategy.batchnorm_decay = old_batchnorm_decay

        return network

    def __pruning(self, net: cv.PyTorchNetwork):
        """
        Procedure for the pruning of the neurons of the PyTorchNetwork passed as an argument.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to prune.

        Returns
        ----------
        PyTorchNetwork
            The PyTorchNetwork resulting from the application of the pure pruning procedure.

        """

        # We transfer the internal pytorch model to the CPU for the pruning procedure.
        net.pytorch_network.cpu()

        # We compute the total number of weights in the batch normalization layers (which, for fully connected networks,
        # is equal to the number of neurons in the corresponding fully-connected layer).
        num_bn_weights = 0
        for m in net.pytorch_network.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                num_bn_weights += m.weight.numel()

        # We copy all the absolute values of the batch norm weights in a new tensor and we sort in ascending order
        bn_weights = torch.zeros(num_bn_weights)
        bn_weights_index = 0
        for m in net.pytorch_network.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                size = m.weight.numel()
                bn_weights[bn_weights_index:(bn_weights_index + size)] = m.weight.abs().clone()
                bn_weights_index += size

        ordered_bn_weights, ordered_bn_indexes = torch.sort(bn_weights)

        # We determine the number of neurons we need to remove, given the sparsity rate.
        threshold_index = math.floor(num_bn_weights * self.sparsity_rate)

        # We select the batch norm weight absolute value we will use as threshold value given the threshold index.
        threshold_value = ordered_bn_weights[threshold_index]

        # We now need to create a new network with the correct number of neurons in the different layers.
        # To do so we assume that in the network after a linear layer there is always a batch norm layer.

        new_layers = []
        previous_layer_mask = None
        old_layers = [m for m in net.pytorch_network.modules()]
        num_layers = len(old_layers)
        for i in range(num_layers):

            if (i == num_layers - 1) and isinstance(old_layers[i], torch.nn.Linear):

                # In this case we are considering the last layer of the network (which we assume to be a linear layer),
                # therefore the number of output of the new layer will be equal to the one of the old layer.

                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                # If the old linear layer had bias then also the new linear layer has them.
                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = int(previous_layer_mask.sum().item())

                # We create the new linear layer with the correct architecture.
                print(num_in_features, old_layers[i].out_features, has_bias)
                new_linear_layer = torch.nn.Linear(num_in_features, old_layers[i].out_features, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias.data

                # We save the new linear layer.
                new_layers.append(new_linear_layer)

            elif isinstance(old_layers[i], torch.nn.Linear) and isinstance(old_layers[i + 1], torch.nn.BatchNorm1d):

                # If the layer old_layers[i] is the first linear layer then the previous layer mask corrspond to the
                # complete input.
                if previous_layer_mask is None:
                    previous_layer_mask = torch.ones(old_layers[i].in_features)

                # We compute the mask corresponding to the batch normalization layer.
                layer_mask = old_layers[i + 1].weight.abs().gt(threshold_value).float()
                new_neuron_number = int(layer_mask.sum().item())

                # We compute the indexes of the non-zero weights for the current batch norm layer and the previous one.
                current_nonzero_indexes = layer_mask.nonzero(as_tuple=True)[0]
                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                # We create the new batch norm layer with the new neuron number.

                new_bn_layer = torch.nn.BatchNorm1d(new_neuron_number, eps=old_layers[i + 1].eps,
                                                    momentum=old_layers[i + 1].momentum,
                                                    affine=old_layers[i + 1].affine,
                                                    track_running_stats=old_layers[i + 1].track_running_stats)

                # We copy the parameters corresponding to the still existing neurons from the old batch norm layer
                # to the new one. They are identified by the indexes in current_nonzero_indexes.

                new_bn_layer.weight.data = old_layers[i + 1].weight[current_nonzero_indexes].clone()
                new_bn_layer.bias.data = old_layers[i + 1].bias[current_nonzero_indexes].clone()
                new_bn_layer.running_mean = old_layers[i + 1].running_mean[current_nonzero_indexes].clone()
                new_bn_layer.running_var = old_layers[i + 1].running_var[current_nonzero_indexes].clone()

                # If the old linear layer had bias then also the new linear layer has them.
                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = int(previous_layer_mask.sum().item())

                # We create the new linear layer with the correct architecture.
                new_linear_layer = torch.nn.Linear(num_in_features, new_neuron_number, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[current_nonzero_indexes, :].clone()
                new_linear_layer.weight.data = new_linear_layer.weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias[current_nonzero_indexes]

                # We save the new layers in the order in which they should be in our sequential model: first the linear
                # layer and then the batch norm layer.
                new_layers.append(new_linear_layer)
                new_layers.append(new_bn_layer)

                # We update the value of previous_layer_mask with the current mask.
                previous_layer_mask = layer_mask

            elif isinstance(old_layers[i], torch.nn.Linear) and not isinstance(old_layers[i + 1], torch.nn.BatchNorm1d):

                # If the layer old_layers[i] is the first linear layer then the previous layer mask correspond to the
                # complete input.
                if previous_layer_mask is None:
                    previous_layer_mask = torch.ones(old_layers[i].in_features)

                # If the linear layer is not followed by a batch normalization layer then it will not be neuron pruned,
                # therefore the layer_mask will be equals to the number of output features of the old layer.
                layer_mask = torch.ones(old_layers[i].out_features)

                # We compute the indexes of the non-zero weights for the current batch norm layer and the previous one.
                current_nonzero_indexes = layer_mask.nonzero(as_tuple=True)[0]
                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = previous_layer_mask.sum().item()

                # We create the new linear layer with the correct architecture.
                new_linear_layer = torch.nn.Linear(num_in_features, old_layers[i].out_features, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[current_nonzero_indexes, :].clone()
                new_linear_layer.weight.data = new_linear_layer.weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias[current_nonzero_indexes]

                # We save the new layers in the order in which they should be in our sequential model: first the linear
                # layer and then the batch norm layer.
                new_layers.append(new_linear_layer)

            elif isinstance(old_layers[i], nn.ReLU):
                new_layers.append(nn.ReLU())

        pruned_network = torch.nn.Sequential(*new_layers)
        net.pytorch_network = pruned_network
        net.identifier = net.identifier + '_pruned'
        return net
