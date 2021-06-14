from enum import Enum

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QRegExp, Qt, QSize
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QDoubleValidator
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QWidget, QLineEdit, QGridLayout, QComboBox, \
    QTextEdit, QPlainTextEdit

import coconet.view.styles as style
import coconet.view.util.utility as u
from coconet.core.controller.pynevertemp.tensor import Tensor
from coconet.core.model.network import NetworkNode
from coconet.view.drawing.element import PropertyBlock, NodeBlock
from coconet.view.widget.loading import ProgressBar

UNEDITABLE = ["weight", "bias", "in_features"]


class ArithmeticValidator:
    """
    This class collects the possible validators for
    the editing dialogs.

    INT : (QIntValidator)
        Integer validator.
    FLOAT : (QDoubleValidator)
        Floating-point validator.
    TENSOR : (QRegExpValidator)
        Tensor ("nxmxl", "nXmXl", "n,m,l") with n, m, l
        integers validator.
    TENSOR_LIST : (QRegExpValidator)
        List of Tensors.

    """

    INT = QIntValidator()
    FLOAT = QDoubleValidator()
    TENSOR = QRegExpValidator(QRegExp("(([0-9])+(,[0-9]+)*)"))
    TENSOR_LIST = QRegExpValidator(QRegExp("(\((([0-9])+(,[0-9]+)*)\))+(,(\((([0-9])+(,[0-9]+)*)\)))*"))


class MessageType(Enum):
    """
    This class collects the different types of messages.

    """

    ERROR = 0
    MESSAGE = 1


class CoCoNetDialog(QtWidgets.QDialog):
    """
    Base class for grouping common elements of the dialogs.
    Each dialog share a main_layout (vertical by default), a title
    and a string content.

    Attributes
    ----------
    layout : QVBoxLayout
        The main main_layout of the dialog.
    title : str
        The dialog title.
    content : str
        The dialog content.

    Methods
    ----------
    render_layout()
        Procedure to update the main_layout.

    """

    def __init__(self, title="CoCoNet Dialog", message="CoCoNet message", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.title = title
        self.content = message

        self.setWindowTitle("\u26a0")
        self.setModal(True)
        self.setStyleSheet("background-color: " + style.GREY_1 + ";")

    def set_title(self, title: str) -> None:
        """
        This method updates the dialog title.

        Parameters
        ----------
        title : str
            New title to display.

        """

        self.title = title

    def set_content(self, content: str) -> None:
        """
        This method updates the dialog content.

        Parameters
        ----------
        content : str
            New content to display.

        """

        self.content = content

    def render_layout(self) -> None:
        """
        This method updates the main_layout with the changes done
        in the child class(es).

        """

        self.setLayout(self.layout)


class MessageDialog(CoCoNetDialog):
    """
    This class is a Dialog displaying a message to the user.

    """

    def __init__(self, message: str, message_type: MessageType):
        super().__init__("", message)

        # Set the dialog stile depending on message_type
        if message_type == MessageType.MESSAGE:
            title_label = QLabel("Message")
            title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        else:
            title_label = QLabel("Error")
            title_label.setStyleSheet(style.ERROR_LABEL_STYLE)

        title_label.setAlignment(Qt.AlignCenter)

        # Set content label
        mess_label = QLabel("\n" + self.content + "\n")
        mess_label.setStyleSheet(style.PARAM_LABEL_STYLE)
        mess_label.setAlignment(Qt.AlignCenter)

        # Add a button to close the dialog
        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(lambda: self.close())
        ok_button.setStyleSheet(style.BUTTON_STYLE)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)
        self.layout.addWidget(ok_button)

        self.render_layout()


class HelpDialog(CoCoNetDialog):
    """
    This dialog displays the user guide from the documentation file.

    """

    def __init__(self):
        super().__init__("User Guide", "User Guide")
        self.setWindowTitle(self.title)
        self.resize(QSize(800, 600))
        self.setStyleSheet("background-color: " + style.GREY_3 + ";")

        # The dialogs contains a text area reading the user guide file
        text = open('docs/coconet/userguide/User Guide.html', encoding="utf8").read()
        text_area = QTextEdit(text)
        text_area.setReadOnly(True)

        self.layout = QHBoxLayout()
        self.layout.addWidget(text_area)
        self.render_layout()


class ConfirmDialog(CoCoNetDialog):
    """
    This dialog asks the user the confirm to clear the
    unsaved workspace before continue.

    Attributes
    ----------
    confirm : bool
        Boolean value to store user decision.

    Methods
    ----------
    ok()
        Register user confirm.
    deny()
        Register user denial.

    """

    def __init__(self, title, message):
        super().__init__(title, message)

        # Set title label
        title_label = QLabel(self.title)
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)

        # Set message label
        mess_label = QLabel("\n" + self.content + "\n")
        mess_label.setStyleSheet(style.PARAM_LABEL_STYLE)
        mess_label.setAlignment(Qt.AlignCenter)

        self.confirm = False

        # Add buttons to close the dialog
        confirm_button = QPushButton("Yes")
        confirm_button.clicked.connect(lambda: self.ok())
        confirm_button.setStyleSheet(style.BUTTON_STYLE)

        no_button = QPushButton("No")
        no_button.clicked.connect(lambda: self.deny())
        no_button.setStyleSheet(style.BUTTON_STYLE)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(confirm_button)
        buttons_layout.addWidget(no_button)
        buttons = QWidget()
        buttons.setLayout(buttons_layout)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)
        self.layout.addWidget(buttons)

        self.render_layout()

    def ok(self) -> None:
        """
        This method sets the result to True and closes the dialog.

        """

        self.confirm = True
        self.close()

    def deny(self) -> None:
        """
        This method sets the result to False and closes the dialog.

        """

        self.confirm = False
        self.close()


class InputDialog(CoCoNetDialog):
    """
    This dialog prompts the user to give an input. After the input
    is validated, it is saved. The input can be a tuple or a
    set of tuples.

    Attributes
    ----------
    input: tuple
        The input representation.

    Methods
    ----------
    save_input()
        Save the input after conversion, and close the dialog.
    cancel()
        Discard the input and close the dialog.

    """

    def __init__(self, message):
        super().__init__("", message)

        # Set title label
        title_label = QLabel("Input required")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)

        # Set message label
        mess_label = QLabel("\n" + self.content + "\n")
        mess_label.setStyleSheet(style.PARAM_LABEL_STYLE)
        mess_label.setAlignment(Qt.AlignCenter)

        # Set input reading
        self.input = None
        input_line = QLineEdit()
        input_line.setStyleSheet(style.VALUE_LABEL_STYLE)
        input_line.setValidator(QRegExpValidator(QRegExp(
            ArithmeticValidator.TENSOR.regExp().pattern() + "|" +
            ArithmeticValidator.TENSOR_LIST.regExp().pattern())))

        # Add buttons to close the dialog
        confirm_button = QPushButton("Ok")
        confirm_button.clicked.connect(lambda: self.save_input())
        confirm_button.setStyleSheet(style.BUTTON_STYLE)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(lambda: self.cancel())
        cancel_button.setStyleSheet(style.BUTTON_STYLE)

        buttons = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(confirm_button)
        buttons_layout.addWidget(cancel_button)
        buttons.setLayout(buttons_layout)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)
        self.layout.addWidget(input_line)
        self.layout.addWidget(buttons)

        self.render_layout()

    def save_input(self) -> None:
        """
        This method saves the input in a tuple and closes the dialog.

        """

        if self.input_line.text() == "":
            self.input = None
        else:
            try:
                self.input = tuple(map(int, self.input_line.text().split(',')))
            except TypeError:
                self.input = None
                error_dialog = MessageDialog("Please check your data format.", MessageType.ERROR)
                error_dialog.show()
        self.close()

    def cancel(self) -> None:
        """
        This method closes the dialog without saving the input read.

        """

        self.input = None
        self.close()


class LoadingDialog(CoCoNetDialog):
    """
    This frameless dialog keeps busy the interface during a
    long action performed by a thread. It shows a message
    and a loading bar.

    """

    def __init__(self, message: str):
        super().__init__("", message)
        # Override window title
        self.setWindowTitle("Wait...")

        # Set content label
        message_label = QLabel(self.content)
        message_label.setStyleSheet(style.LOADING_LABEL_STYLE)

        # Set loading bar
        progress_bar = ProgressBar(self, minimum=0, maximum=0,
                                   textVisible=False, objectName="ProgressBar")
        progress_bar.setStyleSheet(style.PROGRESS_BAR_STYLE)

        # Compose widgets
        self.layout.addWidget(message_label)
        self.layout.addWidget(progress_bar)

        self.render_layout()

        # Disable the dialog frame and close button
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setWindowFlags(Qt.FramelessWindowHint)


class EditSmtPropertyDialog(CoCoNetDialog):
    """
    This dialog allows to define a generic SMT property
    by writing directly in the SMT-LIB language.

    Attributes
    ----------
    property_block : PropertyBlock
        Current property to edit.
    new_property : str
        New SMT-LIB property string.
    smt_box : QPlainTextEdit
        Input box.
    has_edits : bool
        Flag signaling if the property was edited.

    Methods
    ----------
    save_data()
        Procedure to return the new property.

    """

    def __init__(self, property_block: PropertyBlock):
        super().__init__("", "Edit property")
        self.property_block = property_block
        self.new_property = self.property_block.property.property_string
        self.has_edits = False
        self.layout = QGridLayout()

        # Build main_layout
        title_label = QLabel("SMT property")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label, 0, 0, 1, 2)

        # Input box
        smt_label = QLabel("SMT-LIB definition")
        smt_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        smt_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(smt_label, 1, 0)

        self.smt_box = QPlainTextEdit()
        self.smt_box.setStyleSheet(style.VALUE_LABEL_STYLE)
        self.smt_box.insertPlainText(self.new_property)
        self.layout.addWidget(self.smt_box, 1, 1)

        # "Apply" button which saves changes
        apply_button = QPushButton("Apply")
        apply_button.setStyleSheet(style.BUTTON_STYLE)
        apply_button.clicked.connect(lambda: self.save_data())
        self.layout.addWidget(apply_button, 2, 0)

        # "Cancel" button which closes the dialog without saving
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(style.BUTTON_STYLE)
        cancel_button.clicked.connect(lambda: self.close())
        self.layout.addWidget(cancel_button, 2, 1)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)

        self.render_layout()

    def save_data(self):
        self.has_edits = True
        self.new_property = self.smt_box.toPlainText()
        self.close()


class EditPolyhedralPropertyDialog(CoCoNetDialog):
    """
    This dialog allows to define a polyhedral property
    within a controlled environment.

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, property_block: PropertyBlock):
        super().__init__("", "Edit property")
        self.property_block = property_block
        self.new_property = None
        self.has_edits = False
        self.property_list = []
        self.layout = QGridLayout()

        # Build main_layout
        title_label = QLabel("Polyhedral property")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label, 0, 0, 1, 3)

        # Labels
        var_label = QLabel("Variable")
        var_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        var_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(var_label, 1, 0)

        relop_label = QLabel("Operator")
        relop_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        relop_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(relop_label, 1, 1)

        value_label = QLabel("Value")
        value_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        value_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(value_label, 1, 2)

        self.var_cb = QComboBox()
        for v in property_block.variables:
            self.var_cb.addItem(v)
        self.var_cb.setStyleSheet(style.VALUE_LABEL_STYLE)
        self.layout.addWidget(self.var_cb, 2, 0)

        self.op_cb = QComboBox()
        operators = ["<=", "<", ">", ">="]
        for o in operators:
            self.op_cb.addItem(o)
        self.op_cb.setStyleSheet(style.VALUE_LABEL_STYLE)
        self.layout.addWidget(self.op_cb, 2, 1)

        self.val = QLineEdit()
        self.val.setStyleSheet(style.VALUE_LABEL_STYLE)
        self.val.setValidator(ArithmeticValidator.FLOAT)
        self.layout.addWidget(self.val, 2, 2)

        # "Add" button which adds the constraint
        add_button = QPushButton("Add")
        add_button.setStyleSheet(style.BUTTON_STYLE)
        add_button.clicked.connect(
            lambda: self.add_entry(str(self.var_cb.currentText()), str(self.op_cb.currentText()), self.val.text()))
        self.layout.addWidget(add_button, 3, 0)

        # "Save" button which saves the state
        save_button = QPushButton("Save")
        save_button.setStyleSheet(style.BUTTON_STYLE)
        save_button.clicked.connect(lambda: self.save_property())
        self.layout.addWidget(save_button, 3, 1)

        # "Cancel" button which closes the dialog without saving
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(style.BUTTON_STYLE)
        cancel_button.clicked.connect(lambda: self.close())
        self.layout.addWidget(cancel_button, 3, 2)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)

        self.render_layout()

    def add_entry(self, var: str, op: str, val: str):
        constraint = "(assert ("
        end = "))"
        self.property_list.append(f"{constraint}{op} {var} {val}{end}")
        self.var_cb.setCurrentIndex(0)
        self.op_cb.setCurrentIndex(0)
        self.val.setText("")

    def save_property(self):
        self.has_edits = True
        self.add_entry(str(self.var_cb.currentText()),
                       str(self.op_cb.currentText()),
                       self.val.text())

        print(self.property_list)
        self.close()


class EditNodeDialog(CoCoNetDialog):
    """
    This dialog allows to edit the selected node in the canvas.

    Attributes
    ----------
    node_block: NetworkNode
        Current node to edit, which contains information about parameters to
        display and their types.
    parameters: dict
        Dictionary which connects the name of each parameter to its editable
        field, which can be a QLineEdit or a QComboBox.
    edited_data: dict
        Dictionary which contains the edited parameters of the node.
    has_edits: bool
        Parameters that tells if the user has pressed "Apply", so if the
        possible changes to the parameters have to be saved or not.

    Methods
    ----------
    append_node_params(NetworkNode, dict)
        Procedure to display the node parameters in a dialog.
    save_data()
        Procedure to update the values and return.

    """

    def __init__(self, node_block: NodeBlock):
        super().__init__("", "Edit parameters.")
        self.setWindowTitle(node_block.node.name)
        self.layout = QGridLayout()

        # Connect node
        self.node = node_block
        self.parameters = dict()
        self.edited_data = dict()
        self.has_edits = False

        # Build main_layout
        title_label = QLabel("Edit parameters")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label, 0, 0, 1, 2)

        # Input box
        in_dim_label = QLabel("Input")
        in_dim_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        in_dim_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(in_dim_label, 1, 0)

        in_dim_box = QLineEdit()
        in_dim_box.setStyleSheet(style.VALUE_LABEL_STYLE)
        in_dim_box.setText(','.join(map(str, node_block.in_dim)))
        in_dim_box.setValidator(ArithmeticValidator.TENSOR)

        self.layout.addWidget(in_dim_box, 1, 1)
        self.parameters["in_dim"] = in_dim_box

        if not node_block.is_head:
            in_dim_box.setReadOnly(True)

        # Display parameters if present
        counter = 2
        if node_block.node.param:
            counter = self.append_node_params(node_block.node, node_block.block_data)

        # "Apply" button which saves changes
        apply_button = QPushButton("Apply")
        apply_button.setStyleSheet(style.BUTTON_STYLE)
        apply_button.clicked.connect(lambda: self.save_data())
        self.layout.addWidget(apply_button, counter, 0)

        # "Cancel" button which closes the dialog without saving
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(style.BUTTON_STYLE)
        cancel_button.clicked.connect(lambda: self.close())
        self.layout.addWidget(cancel_button, counter, 1)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)

        self.render_layout()

    def append_node_params(self, node: NetworkNode, current_data: dict) -> int:
        """

        This method adds to the dialog layer the editable parameters of
        the node, and returns the last row counter for the grid main_layout.

        Attributes
        ----------
        node: NetworkNode
            The node whose parameters are displayed.
        current_data: dict
            The node current data.
        Returns
        ----------
        int
            The last row counter.

        """

        # Init column counter
        counter = 2

        # Display parameter labels
        for param, value in node.param.items():
            param_label = QLabel(param)
            if param in UNEDITABLE:
                param_label.setStyleSheet(style.UNEDITABLE_PARAM_LABEL_STYLE)
            else:
                param_label.setStyleSheet(style.PARAM_LABEL_STYLE)

            param_label.setAlignment(Qt.AlignRight)
            self.layout.addWidget(param_label, counter, 0)

            # Display parameter values
            if value["type"] == "boolean":
                line = QComboBox()
                line.addItem("True")
                line.addItem("False")
                line.setPlaceholderText(str(current_data[param]))
            else:
                line = QLineEdit()
                if param in UNEDITABLE:
                    line.setReadOnly(True)
                if isinstance(current_data[param], Tensor) or isinstance(current_data[param], np.ndarray):
                    line.setText("(" + ','.join(map(str, current_data[param].shape)) + ")")
                elif isinstance(current_data[param], tuple):
                    line.setText(','.join(map(str, current_data[param])))
                else:
                    line.setText(str(current_data[param]))

                # Set type validator
                validator = None
                if value["type"] == "int":
                    validator = ArithmeticValidator.INT
                elif value["type"] == "float":
                    validator = ArithmeticValidator.FLOAT
                elif value["type"] == "Tensor" or value["type"] == "list of ints":
                    validator = ArithmeticValidator.TENSOR
                elif value["type"] == "list of Tensors":
                    validator = ArithmeticValidator.TENSOR_LIST
                line.setValidator(validator)

            # Set the tooltip of the input with the description
            line.setToolTip("<" + value["type"] + ">: "
                            + value["description"])

            if param in UNEDITABLE:
                line.setStyleSheet(style.UNEDITABLE_VALUE_LABEL_STYLE)
            else:
                line.setStyleSheet(style.VALUE_LABEL_STYLE)
            self.layout.addWidget(line, counter, 1)

            # Keep trace of QLineEdit objects
            self.parameters[param] = line
            counter += 1

        return counter

    def save_data(self) -> None:
        """
        This method saves the changed parameters in their
        correct format, storing them in a dictionary.

        """

        self.has_edits = True

        for key, line in self.parameters.items():
            try:
                if type(line) == QLineEdit:
                    if line.isModified() and len(line.text()) != 0:
                        if key == "in_dim":
                            self.edited_data["in_dim"] = tuple(
                                map(int, line.text().split(',')))
                        else:
                            data_type = self.node.param[key]["type"]

                            # Casting
                            if data_type == "int":
                                self.edited_data[key] = int(line.text())
                            elif data_type == "float":
                                self.edited_data[key] = float(line.text())
                            elif data_type == "Tensor":
                                self.edited_data[key] = u.text_to_tensor(
                                    line.text())
                            elif data_type == "list of ints":
                                self.edited_data[key] = tuple(
                                    map(int, line.text().split(',')))
                            elif data_type == "list of Tensors":
                                self.edited_data[key] = u.text_to_tensor_set(
                                    line.text())

                elif type(line) == QComboBox:
                    if line.currentText() == "True":
                        self.edited_data[key] = True
                    else:
                        self.edited_data[key] = False

            except Exception:
                # If there are errors in data format
                error_dialog = MessageDialog("Please check data format.", MessageType.ERROR)
                error_dialog.show()

        self.close()
