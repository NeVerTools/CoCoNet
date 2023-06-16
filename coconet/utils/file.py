"""
Module file.py

This module contains the classes for handling file open and save

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

import onnx
import pynever.strategies.conversion as conv
import torch
from pynever.networks import NeuralNetwork
from pynever.strategies.smt_reading import ExpressionTreeConverter
from pysmt.exceptions import PysmtException
from pysmt.smtlib.parser import SmtLibParser

from coconet.utils.container import PropertyContainer
from coconet.view.ui.dialog import MessageDialog, MessageType


class FileFormat:
    """
    This class is a container of file formats used in NeVer2

    """

    NETWORK_FORMATS_OPENING = "All supported formats (*.onnx *.pt *.pth);;\
                                ONNX (*.onnx);;\
                                PyTorch (*.pt *.pth)"

    NETWORK_FORMATS_SAVE = "ONNX (*.onnx);;\
                            PyTorch (*.pt *.pth);;\
                            VNNLIB (*.onnx + *.smt2)"

    PROPERTY_FORMATS = "VNN-LIB files (*.smt *.smt2 *.vnnlib);;\
                               SMT (*.smt *.smt2);;\
                               VNNLIB (*.vnnlib)"

    SUPPORTED_NETWORK_FORMATS = {'VNNLIB': ['vnnlib'],
                                 'ONNX': ['onnx'],
                                 'PyTorch': ['pt', 'pth']}

    SUPPORTED_PROPERTY_FORMATS = {'SMT': ['smt', 'smt2'],
                                  'VNNLIB': ['vnnlib']}


def read_variables(property_string: str) -> list:
    """
    This method reads all the variables contained in a constraints string

    Parameters
    ----------
    property_string : str
        A list of constraints contained in a single smt string
    Returns
    -------
    list
        The variables appearing in the constraints

    """

    variables = []

    for line in property_string.split('\n'):
        line = line.replace('(assert (<= ', '').replace('(assert (>= ', '').split(' ')

        if line[0] != '' and line[0] not in variables:
            variables.append(line[0])

    return variables


def read_properties(path: str) -> dict:
    """
    This method reads the SMT property file and
    creates a property for each node.

    Parameters
    ----------
    path : str
        The SMT-LIB file path.

    Returns
    -------
    dict
        The dictionary of properties whose key is the variable_name and value is the stm_string

    """

    parser = SmtLibParser()
    try:
        script = parser.get_script_fname(path)
    except PysmtException:
        dialog = MessageDialog('Failed to parse SMT property.', MessageType.ERROR)
        dialog.exec()
        return dict()

    declarations = script.filter_by_command_name(['declare-fun', 'declare-const'])
    assertions = script.filter_by_command_name('assert')
    var_set = []
    var_list = []
    properties = dict()

    for d in declarations:
        var_list.append(str(d.args[0]).replace('\'', ''))
        varname = str(d.args[0]).split('_')[0].replace('\'', '')  # Variable format is <v_name>_<idx>
        if varname not in var_set:
            var_set.append(varname)

    counter = 0

    for a in assertions:
        line = str(a.args[0]).replace('\'', '')
        for v in var_set:
            if f" {v}" in line or f"({v}" in line:  # Either '(v ...' or '... v)'
                if v not in properties.keys():
                    properties[v] = PropertyContainer()
                    properties[v].smt_string = ''
                    properties[v].variables = list(filter(lambda x: v in x, var_list))
                    counter += 1
                conv = ExpressionTreeConverter()
                wrap = conv.build_from_infix(line).as_prefix()
                properties[v].smt_string += f"(assert {wrap})\n"
                break

    return properties


def write_smt_property(path: str, props: dict, dtype: str) -> None:
    # Create and write file
    with open(path, "w") as f:
        # Variables
        for p in props.values():
            for v in p.variables:
                f.write(f"(declare-const {v} {dtype})\n")
        f.write("\n")

        # Constraints
        for p in props.values():
            f.write(p.smt_string + "\n")


class InputHandler:
    """
    This class provides an interface for reading a file containing a
    NeuralNetwork and converting it in the internal representation.

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self):
        self.extension = ''
        self.alt_repr = None
        self.network_input = None
        self.strategy = None

    def read_network(self, path: str) -> NeuralNetwork:
        """
        This method converts the network from a file to the internal representation.

        Parameters
        ----------
        path : str
            The network file path.

        Returns
        ----------
        NeuralNetwork
            The internal representation of the converted network.

        """

        if path == '':
            raise Exception('Invalid path.')

        # Get extension
        self.extension = path.split('.')[-1]
        net_id = path.split('/')[-1].replace(f'.{self.extension}', '')

        if self.extension in FileFormat.SUPPORTED_NETWORK_FORMATS['ONNX']:
            model_proto = onnx.load(path)
            self.alt_repr = conv.ONNXNetwork(net_id, model_proto, True)

        elif self.extension in FileFormat.SUPPORTED_NETWORK_FORMATS['PyTorch']:
            if not torch.cuda.is_available():
                module = torch.load(path, map_location=torch.device('cpu'))
            else:
                module = torch.load(path)

            self.alt_repr = conv.PyTorchNetwork(net_id, module, True)

        # Convert the network
        if self.alt_repr is None:
            raise Exception('No cached representation')
        else:
            # Converting the network in the internal representation
            # If the chosen format has got an initial input for the network,
            # it is converted in the internal representation
            if isinstance(self.alt_repr, conv.ONNXNetwork):
                self.strategy = conv.ONNXConverter()
            else:
                self.strategy = conv.PyTorchConverter()

            return self.strategy.to_neural_network(self.alt_repr)


class OutputHandler:
    """
    This class converts and saves the network in one of the supported formats.

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self):
        self.extension = None
        self.alt_repr = None
        self.strategy = None

    def save(self, network: NeuralNetwork, filename: tuple) -> None:
        """
        This method converts the current network
        and saves it in the chosen format.

        """

        if '.' not in filename[0]:  # If no explicit extension
            if 'VNNLIB' in filename[1]:
                self.extension = 'vnnlib'
            if self.extension == 'vnnlib':
                filename = (f'{filename[0]}.onnx', filename[1])
            else:
                self.extension = filename[0].split('.')[-1]
                filename = (f'{filename[0]}.{self.extension}', filename[1])
        else:
            self.extension = filename[0].split('.')[-1]

        # The network is converted in the alternative representation
        self.alt_repr = self.convert_network(network, filename[0])

        # Saving the network on file depending on the format
        conv.save_network_path(self.alt_repr, filename[0])

    def save_properties(self, properties: dict, filename: tuple) -> None:
        """
        This method saves the properties in the network as a SMT-LIB
        file. The file shares the same name as the network file, with
        the changed extension.

        Parameters
        ----------
        properties : dict
            The dictionary of defined properties.
        filename : tuple
            The tuple containing the file name and the extension.

        """

        path = filename[0].replace('.' + self.extension, '.smt2')
        if '.' not in path:
            path = path + '.smt2'

        # Update extension
        self.extension = 'smt2'
        write_smt_property(path, properties, 'Real')

    def convert_network(self, network: NeuralNetwork, filename: str) -> conv.AlternativeRepresentation:
        """
        This method converts the internal representation into the chosen
        alternative representation, depending on the extension

        Attributes
        ----------
        network : NeuralNetwork
            The network to convert.
        filename : str
            The file name of the network.

        Returns
        ----------
        AlternativeRepresentation
            The converted network in the required extension.

        """

        # Getting the filename
        net_id = filename.split('/')[-1].split('.')[0]

        if self.extension in FileFormat.SUPPORTED_NETWORK_FORMATS['ONNX'] or \
                self.extension in FileFormat.SUPPORTED_NETWORK_FORMATS['VNNLIB']:
            self.strategy = conv.ONNXConverter()
            self.alt_repr = self.strategy.from_neural_network(network)

        elif self.extension in FileFormat.SUPPORTED_NETWORK_FORMATS['PyTorch']:
            self.strategy = conv.PyTorchConverter()
            self.alt_repr = self.strategy.from_neural_network(network)
        else:
            raise Exception(f'Unsupported format {self.extension}')

        self.alt_repr.identifier = net_id
        self.alt_repr.up_to_date = True
        network.alt_rep_cache.append(self.alt_repr)

        return self.alt_repr
