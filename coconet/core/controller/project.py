import os

import onnx
import coconet.core.controller.pynevertemp.networks as pynn
import torch
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QThread
from PyQt5.QtWidgets import QFileDialog, QApplication
from coconet.core.controller.pynevertemp.strategies.conversion import ONNXNetwork, \
    ONNXConverter, PyTorchConverter, TensorflowConverter, PyTorchNetwork, TensorflowNetwork
from tensorflow import keras

from coconet.view.widget.dialog.dialogs import LoadingDialog, MessageDialog, MessageType, InputDialog

# Formats available for opening and saving networks
FILE_FORMATS_OPENING = "All supported formats(*.onnx *.pt *.pth *.pb);;\
            ONNX(*.onnx);;\
            PyTorch(*.pt *.pth);;\
            TensorFlow(*.pb)"
SUPPORTED_FORMATS = {'ONNX': ['onnx'],
                     'PyTorch': ['pt', 'pth'],
                     'TensorFlow': ['pb']}


class Project(QObject):
    """
    This class manages the opened network by handling file saving and opening,
    and the conversion to/from the internal representation.

    Attributes
    ----------
    file_path : str
        The path of the opened network
    NN : SequentialNetwork
        The current sequential network
    input_handler : InputHandler
        It is instantiated to open and convert a network from a file.
    output_handler : OutputHandler
        It is instantiated to save and convert a network in a file.
    loading_dialog : LoadingDialog
        Dialog to show during the conversions to block the interface.

    Methods
    ----------
    load_network()
        Method called at the end of the conversion to restore the
        interface.
    close_loading_dialog()
        This method closes the loading dialog and restores the cursor.
    save_network()
        Method called at the end of the saving of the network to
        restore the network.
    open()
        This method opens a file and create a thread to read and convert
        it.
    read_input_shape()
        This method gets an input for the network by the user.
    save(bool)
        This method saves a file and create a thread to convert and save
        it.

    """

    # This signal will be connected to the canvas to draw the opened network.
    opened_net = pyqtSignal()

    def __init__(self):
        super(QObject, self).__init__()
        self.file_path = ""
        self.file_name = ("", "")

        self.NN = pynn.SequentialNetwork("")

        self.input_handler = None
        self.output_handler = None

        self.loading_dialog = LoadingDialog("")

    def load_network(self):
        """
        This method loads a network, path and filename,
        closing the loading dialog and restoring the cursor.
        
        """

        # If a network has been read, the signal to draw is is emitted
        if self.input_handler.NN is not None:
            self.NN = self.input_handler.NN
            self.opened_net.emit()
            # Other network data is saved
            self.file_path = self.input_handler.file_path

        # Loading dialog closed and cursor restored
        self.close_loading_dialog()

    def close_loading_dialog(self):
        """
        This method closes the loading dialog and restores the cursor

        """

        self.loading_dialog.close()
        QApplication.restoreOverrideCursor()

    def save_network(self):
        """
        This method saves the opened network, path and filename,
        closing the save dialog and restoring the cursor.

        """

        self.file_path = self.output_handler.file_path

        self.close_loading_dialog()

    def open(self):
        """
        This method opens a file for reading a network in one of the supported
        formats. The network is then converted by a thread, while a
        loading dialog is displayed.

        """

        # Open network
        filepath = QFileDialog.getOpenFileName(None, "Open network", "", FILE_FORMATS_OPENING)

        # If a file has been selected:
        if filepath != ("", ""):
            self.input_handler = InputHandler(filepath[0])

            # A "wait cursor" appears locking the interface
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.loading_dialog = LoadingDialog("Reading network...")

            # Create a thread and connect the end with the proper method
            thread = QThread()
            thread.finished.connect(lambda: self.close_loading_dialog())

            # Move InputHandler to the new thread
            self.input_handler.moveToThread(thread)
            thread.started.connect(lambda: self.input_handler.open())

            self.input_handler.network_read.connect(
                lambda: self.loading_dialog.set_content("Converting network..."))
            self.input_handler.network_converted.connect(lambda: thread.quit())
            self.input_handler.network_converted.connect(lambda: self.input_handler.deleteLater())
            self.input_handler.network_converted.connect(lambda: self.load_network())
            self.input_handler.input_shape_required.connect(lambda: self.read_input_shape())
            self.input_handler.network_not_converted.connect(lambda: self.close_loading_dialog())

            thread.finished.connect(lambda: thread.deleteLater())

            # Start the thread and show the loading dialog
            thread.start()
            self.loading_dialog.exec()

            # At the end of the loading, the main thread looks for potential
            # exceptions in the input_handler
            if self.input_handler.conversion_exception is not None:
                # If there is some error in converting, it is returned
                # an error message
                error_dialog = MessageDialog("Error in network reading: \n"
                                             + str(self.input_handler.conversion_exception),
                                             MessageType.ERROR)
                error_dialog.show()

    def read_input_shape(self):
        """
        This method allows the user to define the input shape of the network
        before opening.

        """

        dialog_text = "Please, give an input shape for the network."
        if self.input_handler.input_exception is not None:
            dialog_text += "\n" + str(self.input_handler.input_exception)

        input_dialog = InputDialog(dialog_text)

        # The cursor is restored to use the dialog
        QApplication.restoreOverrideCursor()
        input_dialog.exec()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if input_dialog.input is not None:
            # Pass input to input_handler and open
            self.input_handler.network_input = input_dialog.input
            self.input_handler.open(input_dialog.input)
        else:
            # Stop the thread and abort loading
            self.input_handler.network_not_converted.emit()

    def save(self, _as: bool = True):
        """
        This method starts a thread for converting the network and saving it.
        It prompts the user before starting.

        Parameters
        ----------
        _as: bool
            This attribute distinguishes between "save" and "save as".
            If _as is True the network will be saved in a new file, while
            if _as is False the network will overwrite the current one.

        """
        # If the user picked "save as" option or there isn't a current file,
        # a dialog is opened to chose where to save the net
        if _as or self.file_name == ("", ""):
            file_name = QFileDialog.getSaveFileName(None, 'Save File', "", FILE_FORMATS_OPENING)
        else:
            # Otherwise the actual name is used to save the net
            file_name = self.file_name

        if file_name != ("", ""):
            self.output_handler = OutputHandler(file_name[0], self.NN)

            # A  "wait cursor" appears locking the interface
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.loading_dialog = LoadingDialog("Converting network...")

            # Create a thread and connect the end with the proper method
            thread = QThread()
            thread.finished.connect(lambda: self.save_network())

            # Move input_handler to the new thread
            self.output_handler.moveToThread(thread)
            thread.started.connect(self.output_handler.save)

            self.output_handler.network_converted.connect(
                lambda: self.loading_dialog.set_content("Saving network..."))
            self.output_handler.network_saved.connect(lambda: thread.quit())
            self.output_handler.network_saved.connect(lambda: self.output_handler.deleteLater())

            thread.finished.connect(lambda: thread.deleteLater())

            # Start the thread and show the loading dialog
            thread.start()
            self.loading_dialog.exec()

            # At the end of the loading, the main thread looks for eventual
            # Exception in the output_handler
            if self.output_handler.exception is not None:
                error_dialog = MessageDialog("Error in network saving: \n"
                                             + str(self.output_handler.exception),
                                             MessageType.ERROR)
                error_dialog.show()
                # If there is some error in converting, None is returned
                # an error message


class InputHandler(QObject):
    """
    This class reads and converts a network of a supported format.

    Attributes
    ----------
    file_path: str
        Complete path of the network.
    extension: str
        Format of the network.
    alt_repr: AlternativeRepresentation
        Network in original format.
    NN: SequentialNetwork
        Network converted in the internal format.
    converter: ConversionStrategy
        Converter from an alternative representation.
    network_input: tuple
        Optional input of the network.
    input_exception: Exception
        Optional exception in the input stage.
    conversion_exception: Exception
        Optional exception in the converting stage.

    Methods
    ----------
    open()
        Procedure to open the network.
    get_network_input_shape()
        Procedure to apply an inout shape.
    read_network()
        Procedure to convert the network in the
        internal representation.

    """

    # Loading success signal
    network_read = pyqtSignal()
    # Conversion success signal
    network_converted = pyqtSignal()
    # Conversion failure signal
    network_not_converted = pyqtSignal()
    # Input shape required signal
    input_shape_required = pyqtSignal()

    def __init__(self, file_path: str):
        super(QObject, self).__init__()
        self.file_path = file_path
        self.extension = ""

        self.alt_repr = None
        self.NN = None

        self.converter = None
        self.network_input = None

        self.input_exception = None
        self.conversion_exception = None

    def open(self, input: tuple = None):
        """
        This method opens a network from a file and converts it into
        the internal representation.

        """

        if self.file_path != "":
            # Get extension
            self.extension = self.file_path.split(".")[-1]

            # Load the network
            self.read_network()
            self.network_read.emit()

            # Convert the network
            if self.alt_repr is not None:
                try:
                    # Converting the network in the internal representation
                    # If the chosen format has got an initial input for the network,
                    # it is converted in the internal representation
                    if isinstance(self.alt_repr, ONNXNetwork):
                        self.converter = ONNXConverter()
                        self.NN = self.converter.to_neural_network(self.alt_repr)
                        # A signal is emitted for the interface to change
                        self.network_converted.emit()
                    else:
                        if input is None:
                            # For the other formats an input is required, so a signal
                            # is emitted for the interface to display a dialog to
                            # get an input from the user
                            self.input_shape_required.emit()
                        else:
                            self.network_input = input
                            self.get_network_input_shape()

                except Exception as e:
                    # Even in case of conversion_exception, the signal of the ending of the
                    # ending is emitted in order to update the interface
                    self.conversion_exception = e
                    self.network_not_converted.emit()
            else:
                self.conversion_exception = Exception("Error in network reading.")
                self.network_not_converted.emit()

    def get_network_input_shape(self):
        """
        This method is called when an input for the network is required, and
        not provided by the original format. The network is converted starting
        from that input.

        """

        try:
            if self.network_input is None:
                raise Exception("Can't open the network without a valid input")

            # Select the format
            if isinstance(self.alt_repr, PyTorchNetwork):
                self.converter = PyTorchConverter()

            elif isinstance(self.alt_repr, TensorflowNetwork):
                self.converter = TensorflowConverter()

            # Converting the network with the given input
            self.NN = self.converter.to_neural_network(self.alt_repr, self.network_input)
            # the finishing signal is emitted for the thread to stop
            self.network_converted.emit()

        except Exception as e:
            # If the input is not correct, an conversion_exception is saved and an interaction
            # with the user is required again
            self.input_exception = e
            self.input_shape_required.emit()

    def read_network(self):
        """
        This method converts the network from the file to an alternative
        representation.

        """

        net_id = self.file_path.split("/")[-1]
        net_id = net_id.split(".")[0]

        if self.extension in SUPPORTED_FORMATS['ONNX']:
            model_proto = onnx.load(self.file_path)
            self.alt_repr = ONNXNetwork(net_id + "_onnx", model_proto, True)

        elif self.extension in SUPPORTED_FORMATS['PyTorch']:
            module = torch.load(self.file_path)
            self.alt_repr = PyTorchNetwork(net_id + "_pytorch", module, True)

        elif self.extension in SUPPORTED_FORMATS['TensorFlow']:
            head = os.path.split(self.file_path)[0]
            module = keras.models.load_model(head)
            self.alt_repr = TensorflowNetwork(net_id + "_tensorflow", module, True)


class OutputHandler(QObject):
    """
    This class converts and saves the network in one of the supported
    formats.

    Attributes
    ----------
    file_path: str
        Absolute path of the network to save
    extension: str
        Extension of the file to save
    alt_repr: AlternativeRepresentation
        Converted network to save
    NN: NeuralNetwork
        Network to convert and save
    exception: Exception
        Exception to handle at the upper level
    converter: ConversionStrategy
        Converter for the chosen format

    Methods
    ----------
    save():
        Saves the network.
    convert_network()
        Converts the network in the desired format.

    """

    # This signal is emitted when the network has been converted
    network_converted = pyqtSignal()
    # This signal is connected to the ending of the thread
    network_saved = pyqtSignal()

    def __init__(self, file_path: str, network: pynn.NeuralNetwork):
        super(QObject, self).__init__()
        self.file_path = file_path
        self.NN = network

        self.extension = None
        self.alt_repr = None

        self.exception = None
        self.converter = None

    def save(self):
        """
        This method converts the current network and saves it in the chosen
        format.

        """

        self.extension = self.file_path.split(".")[-1]

        try:
            # The network is converted in the alternative representation
            self.convert_network()
            # A signal is emitted for the interface to change
            self.network_converted.emit()

            # Saving the network on file depending on the format
            if isinstance(self.alt_repr, ONNXNetwork):
                onnx.save(self.alt_repr.onnx_network.onnx_network, self.file_path)
            elif isinstance(self.alt_repr, PyTorchNetwork):
                torch.save(self.alt_repr.pytorch_network.pytorch_network, self.file_path)
            elif isinstance(self.alt_repr, TensorflowNetwork):
                keras.models.save(self.alt_repr.tensorflow_network, self.file_path)

        except Exception as e:
            self.exception = e

        # A signal is emitted for the interface to change
        self.network_saved.emit()

    def convert_network(self):
        """
        This method converts the internal representation into the chosen
        alternative representation, depending on the extension

        """

        # Getting the filename
        net_id = self.file_path.split("/")[-1]
        net_id = net_id.split(".")[0]

        if self.extension in SUPPORTED_FORMATS['ONNX']:
            self.converter = ONNXConverter()
            model = self.converter.from_neural_network(self.NN)
            self.alt_repr = ONNXNetwork(net_id + "_onnx", model, True)

        elif self.extension in SUPPORTED_FORMATS['PyTorch']:
            self.converter = PyTorchConverter()
            model = self.converter.from_neural_network(self.NN)
            self.alt_repr = PyTorchNetwork(net_id + "_pytorch", model, True)

        elif self.extension in SUPPORTED_FORMATS['TensorFlow']:
            self.converter = TensorflowConverter()
            model = self.converter.from_neural_network(self.NN, self.NN.get_first_node().in_dim)
            self.alt_repr = TensorflowNetwork(net_id + "_tensorflow", model.tensorflow_network, True)
        else:
            raise Exception("Format not supported")

        self.NN.alt_rep_cache.append(self.alt_repr)
