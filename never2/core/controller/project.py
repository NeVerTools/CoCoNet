import os

import onnx
import torch
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QFileDialog, QApplication
from pysmt.smtlib.parser import SmtLibParser
from pysmt.environment import get_env
from tensorflow import keras

import never2.core.controller.pynevertemp.networks as pynn
from never2.core.controller.pynevertemp.strategies.conversion import ONNXNetwork, \
    ONNXConverter, PyTorchConverter, TensorflowConverter, PyTorchNetwork, TensorflowNetwork, AlternativeRepresentation
from never2.view.drawing.element import PropertyBlock
from never2.view.widget.dialog.dialogs import MessageDialog, MessageType, InputDialog

# Formats available for opening and saving networks
NETWORK_FORMATS_OPENING = "All supported formats (*.onnx *.pt *.pth *.pb);;\
            ONNX(*.onnx);;\
            PyTorch(*.pt *.pth);;\
            TensorFlow(*.pb)"
PROPERTY_FORMATS_OPENING = "SMT-LIB files (*.smt *.smt2);;\
                           SMT(*.smt *.smt2)"
SUPPORTED_NETWORK_FORMATS = {'ONNX': ['onnx'],
                             'PyTorch': ['pt', 'pth'],
                             'TensorFlow': ['pb']}
SUPPORTED_PROPERTY_FORMATS = {'SMT': ['smt', 'smt2']}


class Project(QObject):
    """
    This class manages the opened network by handling file saving and opening,
    and the conversion to/from the internal representation.

    Attributes
    ----------
    file_name : (str, str)
        The file name of the network, wrapped in a tuple (name, extension)
    network : NeuralNetwork
        The current sequential network.
    properties : dict
        Dictionary mapping properties to nodes.
    input_handler : InputHandler
        It is instantiated to open and convert a network from a file.
    output_handler : OutputHandler
        It is instantiated to save and convert a network in a file.

    Methods
    ----------
    open()
        This method opens a file converting the network in the internal
        representation.
    save(bool)
        This method saves a file in the desired format.

    """

    # This signal will be connected to the canvas to draw the opened network.
    opened_net = pyqtSignal()
    opened_property = pyqtSignal()

    def __init__(self):
        super(QObject, self).__init__()
        self.file_name = ("", "")

        self.network = pynn.SequentialNetwork("", "")
        self.properties = dict()

        self.input_handler = None
        self.output_handler = None

    def open(self):
        """
        This method opens a file for reading a network in one of the supported
        formats. The network is then converted by a thread, while a
        loading dialog is displayed.

        """

        # Open network
        self.file_name = QFileDialog.getOpenFileName(None, "Open network", "", NETWORK_FORMATS_OPENING)

        # If a file has been selected:
        if self.file_name != ("", ""):
            self.input_handler = InputHandler()

            # A "wait cursor" appears
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.network = self.input_handler.read_network(self.file_name[0])
            QApplication.restoreOverrideCursor()

            # At the end of the loading, the main thread looks for potential
            # exceptions in the input_handler
            if self.input_handler.conversion_exception is not None:
                # If there is some error in converting, it is returned
                # an error message
                error_dialog = MessageDialog("Error in network reading: \n"
                                             + str(self.input_handler.conversion_exception),
                                             MessageType.ERROR)
                error_dialog.show()
            else:
                self.opened_net.emit()

    def open_property(self):
        """
        This method opens a SMT file containing the description
        of the network properties. The file is checked to be
        consistent with the network and then parsed in order
        to build the property objects.

        Returns
        -------

        """
        # Check project
        if not self.network.nodes:
            err = MessageDialog("No network loaded!", MessageType.ERROR)
            err.show()
            return

        # Select file
        property_file_name = QFileDialog.getOpenFileName(None, "Open property file", "", PROPERTY_FORMATS_OPENING)

        if property_file_name != ("", ""):
            self.input_handler = InputHandler()

            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.properties = self.input_handler.read_properties(property_file_name[0])
            QApplication.restoreOverrideCursor()
            self.opened_property.emit()

    def save(self, _as: bool = True):
        """
        This method converts and saves the network in a file.
        It prompts the user before starting.

        Parameters
        ----------
        _as : bool
            This attribute distinguishes between "save" and "save as".
            If _as is True the network will be saved in a new file, while
            if _as is False the network will overwrite the current one.

        """

        # If the user picked "save as" option or there isn't a current file,
        # a dialog is opened to chose where to save the net
        if _as or self.file_name == ("", ""):
            self.file_name = QFileDialog.getSaveFileName(None, 'Save File', "", NETWORK_FORMATS_OPENING)

        if self.file_name != ("", ""):
            self.output_handler = OutputHandler()

            # A  "wait cursor" appears locking the interface
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.output_handler.save(self.network, self.file_name)
            self.output_handler.save_properties(self.properties, self.file_name)
            QApplication.restoreOverrideCursor()

            # At the end of the loading, the main thread looks for eventual
            # Exception in the output_handler
            if self.output_handler.exception is not None:
                error_dialog = MessageDialog("Error in network saving: \n"
                                             + str(self.output_handler.exception),
                                             MessageType.ERROR)
                error_dialog.show()


class InputHandler:
    """
    This class provides an interface for reading a file containing a
    NeuralNetwork and converting it in the internal representation.

    Attributes
    ----------
    extension : str
        Format of the network.
    alt_repr : AlternativeRepresentation
        Network in original format.
    network_input : tuple
        Optional input of the network.
    strategy : ConversionStrategy
        Converter from an alternative representation.
    input_exception : Exception
        Optional exception in the input stage.
    conversion_exception : Exception
        Optional exception in the conversion stage.

    Methods
    ----------
    read_network(str)
        Procedure to read and convert the network in the
        internal representation.
    read_input_dialog()
        Procedure to open a dialog and get an input for
        the network.
    set_input_shape(tuple)
        Procedure to apply an input shape.

    """

    def __init__(self):
        self.extension = ""
        self.alt_repr = None
        self.network_input = None
        self.strategy = None

        self.input_exception = None
        self.conversion_exception = None

    def read_network(self, path: str) -> pynn.NeuralNetwork:
        """
        This method converts the network from the file to an alternative
        representation.

        Parameters
        ----------
        path : str
            The network file path.

        Returns
        ----------
        NeuralNetwork
            The internal representation of the converted network.

        """

        if path == "":
            raise Exception("Invalid path.")

        # Get extension
        self.extension = path.split(".")[-1]
        net_id = path.split("/")[-1].split(".")[0]

        if self.extension in SUPPORTED_NETWORK_FORMATS['ONNX']:
            model_proto = onnx.load(path)
            self.alt_repr = ONNXNetwork(net_id + "_onnx", model_proto, True)

        elif self.extension in SUPPORTED_NETWORK_FORMATS['PyTorch']:
            module = torch.load(path)
            self.alt_repr = PyTorchNetwork(net_id + "_pytorch", module, True)

        elif self.extension in SUPPORTED_NETWORK_FORMATS['TensorFlow']:
            head = os.path.split(path)[0]
            module = keras.models.load_model(head)
            # self.alt_repr = TensorflowNetwork(net_id + "_tensorflow", module, True)
            self.alt_repr = TensorflowNetwork(net_id + "_tensorflow", True)

        # Convert the network
        if self.alt_repr is not None:
            try:
                # Converting the network in the internal representation
                # If the chosen format has got an initial input for the network,
                # it is converted in the internal representation
                if isinstance(self.alt_repr, ONNXNetwork):
                    self.strategy = ONNXConverter()
                    return self.strategy.to_neural_network(self.alt_repr)
                else:
                    self.network_input = self.read_input_dialog()
                    return self.set_input_shape(self.network_input)

            except Exception as e:
                # Even in case of conversion_exception, the signal of the ending of the
                # ending is emitted in order to update the interface
                self.conversion_exception = e
        else:
            self.conversion_exception = Exception("Error in network reading.")

    def read_properties(self, path: str) -> dict:
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
            The dictionary of properties

        """

        parser = SmtLibParser()
        script = parser.get_script_fname(path)
        declarations = script.filter_by_command_name(['declare-fun', 'declare-const'])
        assertions = script.filter_by_command_name('assert')
        var_list = []
        properties = dict()

        for d in declarations:
            varname = str(d.args[0]).split('_')[0].replace('\'', '')  # Variable format is <v_name>_<idx>
            if varname not in var_list:
                var_list.append(varname)

        counter = 0

        for a in assertions:
            line = str(a.args[0]).replace('\'', '')
            tokens = line.replace('(', '').replace(')', '').split()
            for v in var_list:
                if f" {v}" in line or f"({v}" in line:  # Either '(v ...' or '... v)'
                    if v not in properties.keys():
                        properties[v] = PropertyBlock(f"{counter}Pr", "Generic SMT")
                        properties[v].smt_string = ''
                        counter += 1
                    wrap = '(assert ' + f"({tokens[1]} {tokens[0]} {tokens[2]})" + ')'
                    properties[v].smt_string += f"{wrap}\n"
                    break

        return properties

    def read_input_dialog(self) -> tuple:
        """
        This method allows the user to define the input shape of the network
        before opening.

        Returns
        ----------
        tuple
            The input shape given by the user.

        """

        dialog_text = "Please, give an input shape for the network."
        if self.input_exception is not None:
            dialog_text += "\n" + str(self.input_exception)

        input_dialog = InputDialog(dialog_text)

        # The cursor is restored to use the dialog
        QApplication.restoreOverrideCursor()
        input_dialog.exec()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if input_dialog.input is not None:
            # Pass input to input_handler and open
            return input_dialog.input

    def set_input_shape(self, input_shape: tuple) -> pynn.NeuralNetwork:
        """
        This method is called when an input for the network is required, and
        not provided by the original format. The network is converted starting
        from that input.

        Parameters
        ----------
        input_shape : tuple
            The provided shape.

        Returns
        ----------
        NeuralNetwork
            The network with the new input shape.

        """

        try:
            if input_shape is None:
                raise Exception("Can't open the network without a valid input")

            # Select the format
            if isinstance(self.alt_repr, PyTorchNetwork):
                self.strategy = PyTorchConverter()

            elif isinstance(self.alt_repr, TensorflowNetwork):
                self.strategy = TensorflowConverter()

            # Converting the network with the given input
            # TODO ASSOCIATE NEW INPUT SHAPE?
            return self.strategy.to_neural_network(self.alt_repr)

        except Exception as e:
            # If the input is not correct, an conversion_exception is saved and an interaction
            # with the user is required again
            self.input_exception = e


class OutputHandler:
    """
    This class converts and saves the network in one of the supported
    formats.

    Attributes
    ----------
    extension : str
        Extension of the network to save.
    alt_repr : AlternativeRepresentation
        Converted representation of the network
        to save.
    exception : Exception
        Exception to handle at the upper level
    strategy : ConversionStrategy
        Converter for the chosen format

    Methods
    ----------
    save(NeuralNetwork, tuple)
        Saves the network with the given name and format.
    convert_network()
        Converts the network in the desired format.

    """

    def __init__(self):
        self.extension = None
        self.alt_repr = None
        self.exception = None
        self.strategy = None

    def save(self, network: pynn.NeuralNetwork, filename: tuple) -> None:
        """
        This method converts the current network and saves it in the chosen
        format.

        """

        self.extension = filename[1].split(".")[-1].replace(")", "")

        try:
            # The network is converted in the alternative representation
            self.alt_repr = self.convert_network(network, filename[0])

            # Saving the network on file depending on the format
            if isinstance(self.alt_repr, ONNXNetwork):
                onnx.save(self.alt_repr.onnx_network.onnx_network, filename[0])
            elif isinstance(self.alt_repr, PyTorchNetwork):
                torch.save(self.alt_repr.pytorch_network.pytorch_network, filename[0])
            elif isinstance(self.alt_repr, TensorflowNetwork):
                keras.models.save(self.alt_repr, filename)

        except Exception as e:
            self.exception = e

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

        path = filename[0].replace("." + self.extension, ".smt2")

        # Update extension
        self.extension = "smt2"

        # Create and write file
        with open(path, "w") as f:
            # Variables
            for p in properties.values():
                for v in p.variables:
                    f.write("(declare-const " + v + " Real)\n")
            f.write("\n")

            # Constraints
            for p in properties.values():
                f.write(p.smt_string + "\n")

    def convert_network(self, network: pynn.NeuralNetwork, filename: str) -> AlternativeRepresentation:
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
        net_id = filename.split("/")[-1]

        if self.extension in SUPPORTED_NETWORK_FORMATS['ONNX']:
            self.strategy = ONNXConverter()
            model = self.strategy.from_neural_network(network)
            self.alt_repr = ONNXNetwork(net_id + "_onnx", model, True)

        elif self.extension in SUPPORTED_NETWORK_FORMATS['PyTorch']:
            self.strategy = PyTorchConverter()
            model = self.strategy.from_neural_network(network)
            self.alt_repr = PyTorchNetwork(net_id + "_pytorch", model, True)

        elif self.extension in SUPPORTED_NETWORK_FORMATS['TensorFlow']:
            self.strategy = TensorflowConverter()
            model = self.strategy.from_neural_network(network)
            self.alt_repr = TensorflowNetwork(net_id + "_tensorflow", True)
        else:
            raise Exception("Format not supported")

        network.alt_rep_cache.append(self.alt_repr)  # TODO ?
        return self.alt_repr
