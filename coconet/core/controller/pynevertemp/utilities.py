import coconet.core.controller.pynevertemp.strategies.conversion as cv
import coconet.core.controller.pynevertemp.nodes as nodes
import coconet.core.controller.pynevertemp.networks as networks
import coconet.core.controller.pynevertemp.datasets as datasets
import torch
import math
import torch.nn.functional as funct
from coconet.core.controller.pynevertemp.tensor import Tensor
import numpy as np


def combine_batchnorm1d(linear: nodes.FullyConnectedNode, batchnorm: nodes.BatchNormNode) -> nodes.FullyConnectedNode:
    """
    Utility function to combine a BatchNormNode node with a FullyConnectedNode in a corresponding FullyConnectedNode.

    Parameters
    ----------
    linear : FullyConnectedNode
        FullyConnectedNode to combine.
    batchnorm : BatchNormNode
        BatchNorm1DNode to combine.

    Return
    ----------
    FullyConnectedNode
        The FullyConnectedNode resulting from the fusion of the two input nodes.

    """

    l_weight = torch.from_numpy(linear.weight)
    l_bias = torch.from_numpy(linear.bias)
    bn_running_mean = torch.from_numpy(batchnorm.running_mean)
    bn_running_var = torch.from_numpy(batchnorm.running_var)
    bn_weight = torch.from_numpy(batchnorm.weight)
    bn_bias = torch.from_numpy(batchnorm.bias)
    bn_eps = batchnorm.eps

    fused_bias = torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps))
    fused_bias = torch.mul(fused_bias, torch.sub(l_bias, bn_running_mean))
    fused_bias = torch.add(fused_bias, bn_bias)

    fused_weight = torch.diag(torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps)))
    fused_weight = torch.matmul(fused_weight, l_weight)

    fused_linear = nodes.FullyConnectedNode(linear.identifier, linear.in_features, linear.out_features, fused_weight.numpy(),
                                            fused_bias.numpy())

    return fused_linear


def combine_batchnorm1d_net(network: networks.SequentialNetwork) -> networks.SequentialNetwork:
    """
    Utilities function to combine all the FullyConnectedNodes followed by BatchNorm1DNodes in corresponding
    FullyConnectedNodes.

    Parameters
    ----------
    network : SequentialNetwork
        Sequential Network of interest of which we want to combine the nodes.

    Return
    ----------
    SequentialNetwork
        Corresponding Sequential Network with the combined nodes.

    """

    if not network.up_to_date:

        for alt_rep in network.alt_rep_cache:

            if alt_rep.up_to_date:

                if isinstance(alt_rep, cv.PyTorchNetwork):
                    pytorch_cv = cv.PyTorchConverter()
                    network = pytorch_cv.to_neural_network(alt_rep)
                elif isinstance(alt_rep, cv.ONNXNetwork):
                    onnx_cv = cv.ONNXConverter
                    network = onnx_cv.to_neural_network(alt_rep)
                else:
                    raise NotImplementedError
                break

    combined_network = networks.SequentialNetwork(network.identifier + '_combined')

    current_node = network.get_first_node()
    node_index = 1
    while network.get_next_node(current_node) is not None and current_node is not None:

        next_node = network.get_next_node(current_node)
        if isinstance(current_node, nodes.FullyConnectedNode) and isinstance(next_node, nodes.BatchNorm1DNode):
            combined_node = combine_batchnorm1d(current_node, next_node)
            combined_node.identifier = f"Combined_Linear_{node_index}"
            combined_network.add_node(combined_node)
            next_node = network.get_next_node(next_node)

        elif isinstance(current_node, nodes.FullyConnectedNode):
            identifier = f"Linear_{node_index}"
            new_node = nodes.FullyConnectedNode(identifier, current_node.in_features, current_node.out_features,
                                                current_node.weight, current_node.bias)
            combined_network.add_node(new_node)

        elif isinstance(current_node, nodes.ReLUNode):
            identifier = f"ReLU_{node_index}"
            new_node = nodes.ReLUNode(identifier, current_node.num_features)
            combined_network.add_node(new_node)
        else:
            raise NotImplementedError

        node_index += 1
        current_node = next_node

    if isinstance(current_node, nodes.FullyConnectedNode):
        identifier = f"Linear_{node_index}"
        new_node = nodes.FullyConnectedNode(identifier, current_node.in_features, current_node.out_features,
                                            current_node.weight, current_node.bias)
        combined_network.add_node(new_node)
    elif isinstance(current_node, nodes.ReLUNode):
        identifier = f"ReLU_{node_index}"
        new_node = nodes.ReLUNode(identifier, current_node.num_features)
        combined_network.add_node(new_node)
    else:
        raise NotImplementedError

    return combined_network


def testing(net: cv.PyTorchNetwork, dataset: datasets.Dataset, test_batch_size: int, cuda: bool) -> (float, float):
    """
    Testing procedure for a PyTorchNetwork.

    Parameters
    ----------
    net : PyTorchNetwork
        Neural Network to test.
    dataset : Dataset
        Dataset used for testing the network.
    test_batch_size : int
        Dimension for the test batch size for the testing procedure
    cuda : bool
        Whether to use the cuda library for the procedure (default: False).

    Returns
    ----------
    (float, float)
        Rate of correct samples and loss.

    """

    if cuda:
        net.pytorch_network.cuda()
    else:
        net.pytorch_network.cpu()

    test_set = dataset.get_test_set()

    net.pytorch_network.eval()
    net.pytorch_network.float()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        batch_idx = 0
        data_idx = 0

        while data_idx < len(test_set[0]):

            if data_idx + test_batch_size >= len(test_set[0]):
                last_data_idx = len(test_set[0])
            else:
                last_data_idx = data_idx + test_batch_size

            data = torch.from_numpy(test_set[0][data_idx:last_data_idx, :])
            target = torch.from_numpy(test_set[1][data_idx:last_data_idx])

            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
            output = net.pytorch_network(data)
            test_loss += funct.cross_entropy(output, target, reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            batch_idx += 1
            data_idx += test_batch_size

    test_loss /= float(math.floor(len(test_set[0]) / test_batch_size))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_set[0]),
        100. * correct / len(test_set[0])))

    return correct / float(len(test_set[0])), test_loss


def generate_targeted_linf_robustness_query(data: Tensor, adv_target: int, bounds: tuple,
                                            num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate a targeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        for i in range(num_classes):

            if i != adv_target:
                f.write(f"(assert (<= (- Y_{i} Y_{adv_target}) 0))\n")

def generate_untargeted_linf_robustness_query(data: Tensor, target: int, bounds: tuple,
                                              num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate an untargeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        output_query = "(assert (or"
        for i in range(num_classes):

            if i != target:
                output_query += f" (<= (- Y_{target} Y_{i}) 0)"

        output_query += "))"
        f.write(output_query)


def parse_linf_robustness_smtlib(filepath: str) -> (bool, list, int):
    """
    Function to extract the parameters of a robustness query from the smtlib file.
    It assume the SMTLIB file is structured as following:

        ; definition of the variables of interest
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        ...
        (declare-const Y_1 Real)
        (declare-const Y_2 Real)
        ...
        ; definition of the constraints
        (assert (>= X_0 eps_0))
        (assert (<= X_0 eps_1))
        ...
        (assert (<= (- Y_0 Y_1) 0))
        ...

    Where the eps_i are Real numbers.

    Parameters
    ----------
    filepath : str
        Filepath to the SMTLIB file.

    Returns
    ----------
    (bool, list, int)
        Tuple of list: the first list contains the values eps_i for each variables as tuples (lower_bound, upper_bound),
        while the int correspond to the desired target for the related data.
    """
    targeted = True
    correct_target = -1
    lb = []
    ub = []
    with open(filepath, 'r') as f:

        for line in f:

            line = line.replace('(', '( ')
            line = line.replace(')', ' )')
            if line[0] == '(':
                aux = line.split()
                if aux[1] == 'assert':

                    if aux[4] == '(':
                        if aux[3] == 'or':
                            targeted = False
                            temp = aux[8].split("_")
                            correct_target = int(temp[1])
                        else:
                            targeted = True
                            temp = aux[7].split("_")
                            correct_target = int(temp[1])

                    else:

                        if aux[3] == ">=":
                            lb.append(float(aux[5]))
                        else:
                            ub.append(float(aux[5]))

    input_bounds = []
    for i in range(len(lb)):
        input_bounds.append((lb[i], ub[i]))

    return targeted, input_bounds, correct_target


def net_update(network: networks.NeuralNetwork) -> networks.NeuralNetwork:

    if not network.up_to_date:

        for alt_rep in network.alt_rep_cache:

            if alt_rep.up_to_date:
                if isinstance(alt_rep, cv.ONNXNetwork):
                    return cv.ONNXConverter().to_neural_network(alt_rep)
                elif isinstance(alt_rep, cv.PyTorchNetwork):
                    return cv.PyTorchConverter().to_neural_network(alt_rep)
                else:
                    raise NotImplementedError

    else:
        return network


def parse_acas_property(filepath: str) -> ((Tensor, Tensor), (Tensor, Tensor)):

    in_coeff = np.zeros((10, 5))
    in_bias = np.zeros((10, 1))
    out_coeff = []
    out_bias = []
    row_index = 0

    with open(filepath, 'r') as f:

        for line in f:

            if line[0] == "x":
                splitted_line = line.split(" ")
                var_index = int(splitted_line[0][1])
                if splitted_line[1] == ">=":
                    in_coeff[row_index, var_index] = -1
                    in_bias[row_index] = -float(splitted_line[2])
                else:
                    in_coeff[row_index, var_index] = 1
                    in_bias[row_index] = float(splitted_line[2])

            else:

                splitted_line = line.split(" ")
                if len(splitted_line) == 3:
                    var_index = int(splitted_line[0][1])
                    temp = np.zeros(5)
                    if splitted_line[1] == ">=":
                        temp[var_index] = -1
                        out_coeff.append(temp)
                        out_bias.append(-float(splitted_line[2]))
                    else:
                        temp[var_index] = 1
                        out_coeff.append(temp)
                        out_bias.append(float(splitted_line[2]))
                else:
                    var_index_1 = int(splitted_line[0][2])
                    var_index_2 = int(splitted_line[1][2])
                    temp = np.zeros(5)
                    if splitted_line[2] == ">=":
                        temp[var_index_1] = -1
                        temp[var_index_2] = 1
                        out_coeff.append(temp)
                        out_bias.append(-float(splitted_line[3]))
                    else:
                        temp[var_index_1] = 1
                        temp[var_index_2] = -1
                        out_coeff.append(temp)
                        out_bias.append(float(splitted_line[3]))

            row_index = row_index + 1

        out_coeff = np.array(out_coeff)
        array_out_bias = np.zeros((len(out_bias), 1))

        for i in range(len(out_bias)):
            array_out_bias[i, 0] = out_bias[i]

        out_bias = array_out_bias

    return (in_coeff, in_bias), (out_coeff, out_bias)


def parse_nnet(filepath: str) -> (list, list, list, list, list, list):

    with open(filepath) as f:

        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line = f.readline()
            cnt += 1
        # numLayers does't include the input layer!
        numLayers, inputSize, outputSize, maxLayersize = [int(x) for x in line.strip().split(",")[:-1]]
        line = f.readline()

        # input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]

        weights = []
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]
            # weights
            weights.append([])
            biases.append([])
            # weights
            for i in range(currentLayerSize):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                weights[layernum].append([])
                for j in range(previousLayerSize):
                    weights[layernum][i].append(aux[j])
            # biases
            for i in range(currentLayerSize):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum].append(x)

        numLayers = numLayers
        layerSizes = layerSizes
        inputSize = inputSize
        outputSize = outputSize
        maxLayersize = maxLayersize
        inputMinimums = inputMinimums
        inputMaximums = inputMaximums
        inputMeans = means[:-1]
        inputRanges = ranges[:-1]
        outputMean = means[-1]
        outputRange = ranges[-1]
        weights = weights
        biases = biases

        new_weights = []
        new_biases = []
        for i in range(numLayers):
            weight = np.array(weights[i])
            bias = np.array(biases[i])

            new_weights.append(weight)
            new_biases.append(bias)

        return new_weights, new_biases, inputMeans, inputRanges, outputMean, outputRange


def input_search(net: networks.NeuralNetwork, ref_output: Tensor, start_input: Tensor, max_iter: int, rate: float,
                 threshold: float = 1e-5):

    py_net = cv.PyTorchConverter().from_neural_network(net).pytorch_network
    py_ref_output = torch.from_numpy(ref_output)
    py_start_input = torch.from_numpy(start_input)
    current_input = py_start_input
    current_input.requires_grad = True

    optim = torch.optim.SGD(params=[current_input], lr=rate)

    real_output = py_net(current_input)
    py_ref_output = torch.unsqueeze(py_ref_output, 0)
    real_output = torch.unsqueeze(real_output, 0)
    dist = funct.pairwise_distance(real_output, py_ref_output, p=2)
    iteration = 0

    while dist > threshold and iteration < max_iter:
        #current_input.requires_grad = True
        optim.zero_grad()
        dist.backward()
        optim.step()
        real_output = py_net(current_input)
        real_output = torch.unsqueeze(real_output, 0)
        dist = funct.pairwise_distance(py_ref_output, real_output, p=2)
        print(f"Loss: {dist}")
        print(f"Current Input: {current_input}")
        print(f"Current Output: {real_output}")
        print(f"Grad: {current_input.grad}")
        print(f"Ref Output: {ref_output}")
        iteration = iteration + 1

    correct = False
    if dist <= threshold:
        correct = True

    return correct, current_input


def compute_saliency(net: networks.NeuralNetwork, ref_input: Tensor):

    class BackHook:

        def __init__(self, module: torch.nn.Module, backward=True):
            if backward:
                self.hook = module.register_backward_hook(self.hook_fn)
            else:
                self.hook = module.register_forward_hook(self.hook_fn)
            self.m_input = None
            self.m_output = None

        def hook_fn(self, module, m_input, m_output):
            self.m_input = m_input
            self.m_output = m_output

        def close(self):
            self.hook.remove()

    py_net = cv.PyTorchConverter().from_neural_network(net).pytorch_network

    # We register the hooks on the modules of the networks
    backward_hooks = [BackHook(layer) for layer in py_net.modules()]
    forward_hooks = [BackHook(layer, False) for layer in py_net.modules()]

    ref_input = torch.from_numpy(ref_input)
    ref_input.requires_grad = True
    out = py_net(ref_input)
    i = 0
    print("FORWARD HOOKS")
    for m in py_net.modules():
        hook = forward_hooks[i]
        print(m)
        print("INPUT")
        print(hook.m_input)
        print("OUTPUT")
        print(hook.m_output)
        i = i + 1
    for k in range(len(out)):
        print(f"Variable {k} of output")
        out = py_net(ref_input)
        out[k].backward(retain_graph=True)
        print("INPUT GRAD:" + f"{ref_input.grad}")

        i = 0
        for m in py_net.modules():
            hook = backward_hooks[i]
            print(m)
            print("INPUT")
            print(hook.m_input[0])
            print("OUTPUT")
            print(hook.m_output[0])
            i = i + 1

    print(out)
    ref_input[0] = ref_input[0] + 10
    out = py_net(ref_input)
    print(out)