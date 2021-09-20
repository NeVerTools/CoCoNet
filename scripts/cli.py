import os

import onnx
import pynever
import torch
from pynever.networks import SequentialNetwork
from pynever.strategies.conversion import PyTorchNetwork, PyTorchConverter, ONNXConverter, ONNXNetwork

SUPPORTED_NETWORK_FORMATS = {'VNNLIB': ['vnnlib'],
                             'ONNX': ['onnx'],
                             'PyTorch': ['pt', 'pth']}


def check_vnnlib_compliance(model_path: str) -> bool:
    """
    This method performs the VNN-LIB compliance check without the GUI
    in order to be executed as a script.

    Parameters
    ----------
    model_path : str
        The path to the model to be checked.

    Returns
    -------
    bool
        True if the network is VNN-LIB compliant, False otherwise.

    """

    nn = os.path.abspath(model_path)

    if not os.path.isfile(nn):
        print("Invalid model path.")
        return False
    else:
        extension = nn.split(".")[-1]
        net_id = nn.split("/")[-1].split(".")[0]
        alt_repr = None

        if extension in SUPPORTED_NETWORK_FORMATS['ONNX']:
            model_proto = onnx.load(nn)
            alt_repr = ONNXNetwork(net_id, model_proto, True)

        elif extension in SUPPORTED_NETWORK_FORMATS['PyTorch']:
            module = torch.load(nn)
            alt_repr = PyTorchNetwork(net_id, module, True)

        # Convert the network
        if alt_repr is not None:
            try:
                # Converting the network in the internal representation
                # If the chosen format has got an initial input for the network,
                # it is converted in the internal representation
                if isinstance(alt_repr, ONNXNetwork):
                    strategy = ONNXConverter()
                else:
                    strategy = PyTorchConverter()

                network = strategy.to_neural_network(alt_repr)
                if isinstance(network, pynever.networks.SequentialNetwork):
                    print("This network is VNN-LIB compliant")
                    return True
                else:
                    print("This network is not VNN-LIB compliant")
                    return False

            except Exception as e:
                # Even in case of conversion_exception, the signal of the ending of the
                # ending is emitted in order to update the interface
                print("This network is not VNN-LIB compliant.")
                print(e)
                return False
        else:
            print("This network is not VNN-LIB compliant")
            return False


def convert_to_onnx(model_path: str) -> bool:
    """
    This method performs the network conversion without the GUI
    in order to be executed as a script.

    Parameters
    ----------
    model_path : str
        The path to the model to be converted.

    Returns
    -------
    bool
        True if the conversion was successful, False otherwise.

    """

    nn_path = os.path.abspath(model_path)

    if not os.path.isfile(nn_path):
        print("Invalid model path.")
        return False
    else:
        extension = nn_path.split(".")[-1]
        net_id = nn_path.split("/")[-1].split(".")[0]

        if extension in SUPPORTED_NETWORK_FORMATS['ONNX']:
            print("The network is already in the ONNX format.")
            return True

        elif extension in SUPPORTED_NETWORK_FORMATS['PyTorch']:
            module = torch.load(nn_path)
            alt_repr = PyTorchNetwork(net_id, module, True)

        else:
            print("Conversion supported only for PyTorch networks.")
            return False

        if alt_repr is not None:
            try:
                # If the chosen format has got an initial input for the network,
                # it is converted in the internal representation
                strategy = PyTorchConverter()
                network = strategy.to_neural_network(alt_repr)
                if not isinstance(network, SequentialNetwork):
                    print("The network is not VNN-LIB compliant, only sequential networks are supported.")
                    return False

                new_id = net_id + "_converted"
                strategy = ONNXConverter()
                model = strategy.from_neural_network(network)
                onnx_net = ONNXNetwork(new_id, model, True)
                old_ext = '.' + extension
                onnx.save(onnx_net.onnx_network.onnx_network, nn_path.replace(old_ext, '_converted.onnx'))

                print("Conversion successful.")
                return True

            except Exception as e:
                print("The network is not VNN-LIB compliant.")
                print(e)
                return False

        else:
            print("There was an unexpected error.")
            return False


def help():
    print("usage: coconet ... [-check | -convert] [arg]")
    print("Options and arguments:")
    print("no args      : launch CoCoNet in GUI mode.")
    print("-check arg   : checks if the model passed via arg is compliant")
    print("               with the VNN-LIB standard.")
    print("-convert arg : converts the model passed via arg to the ONNX")
    print("               format.")
    print()
    print("arg ...      : arguments passed to program in sys.argv[1:]")
    print()
