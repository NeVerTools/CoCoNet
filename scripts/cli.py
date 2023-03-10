import os

import pynever.networks as nets
import pynever.strategies.conversion as conv


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

    nn_path = os.path.abspath(model_path)

    if not os.path.isfile(nn_path):
        print("Invalid model path.")
        return False
    else:
        # Read the network file
        alt_repr = conv.load_network_path(nn_path)

        # Convert the network
        if alt_repr is not None:
            try:
                if isinstance(alt_repr, conv.ONNXNetwork):
                    strategy = conv.ONNXConverter()
                elif isinstance(alt_repr, conv.PyTorchNetwork):
                    strategy = conv.PyTorchConverter()
                else:
                    strategy = conv.TensorflowConverter()

                network = strategy.to_neural_network(alt_repr)

                # Check compliance
                if isinstance(network, nets.SequentialNetwork):
                    print("This network is VNN-LIB compliant")
                    return True
                else:
                    print("This network is not VNN-LIB compliant")
                    return False

            except Exception as e:
                print(f"Error reading network: {e}")
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
        # Read the network file
        alt_repr = conv.load_network_path(nn_path)
        extension = nn_path.split(".")[-1]

        # Convert the network
        if alt_repr is not None:
            try:
                if isinstance(alt_repr, conv.ONNXNetwork):
                    print('The network is already in the ONNX format.')
                    return True

                elif isinstance(alt_repr, conv.PyTorchNetwork):
                    strategy = conv.PyTorchConverter()
                else:
                    strategy = conv.TensorflowConverter()

                network = strategy.to_neural_network(alt_repr)

                if network is not None and isinstance(network, nets.SequentialNetwork):
                    new_id = network.identifier + '_converted'
                    model = conv.ONNXConverter().from_neural_network(network)
                    onnx_net = conv.ONNXNetwork(new_id, model, True)
                    old_ext = '.' + extension
                    conv.save_network_path(onnx_net, nn_path.replace(old_ext, '_converted.onnx'))

                    print('Conversion successful')
                    return True
                else:
                    print('Conversion supported only for PyTorch and TensorFlow networks.')
                    return False

            except Exception as e:
                print(f"Conversion error: {e}")
        else:
            print('There was an unexpected error reading the model file.')
            return False


def show_help():
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
