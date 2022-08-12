from enum import Enum

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRegExp, Qt, QSize
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QDoubleValidator
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QTextEdit
from pynever.networks import SequentialNetwork
from pynever.tensor import Tensor

import coconet.view.styles as style
import coconet.view.util.utility as u
from coconet import ROOT_DIR
from coconet.core.model.network import NetworkNode
from coconet.view.drawing.element import PropertyBlock, NodeBlock
from coconet.view.widget.custom import CustomButton, CustomTextArea, CustomTextBox, CustomComboBox, CustomLabel


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
    SAMPLE = QRegExpValidator(QRegExp('^(?:\d+(?:\.\d*)?|\.\d+)(?:,(?:\d+(?:\.\d*)?|\.\d+))*$'))


class MessageType(Enum):
    """
    This class collects the different types of messages.

    """

    ERROR = 0
    MESSAGE = 1


class CoCoNetDialog(QtWidgets.QDialog):
    """
    Base class for grouping common elements of the dialogs.
    Each dialog shares a main_layout (vertical by default), a title
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

    def __init__(self, title='CoCoNet Dialog', message='CoCoNet message', parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.title = title
        self.content = message

        if self.title == "":
            self.setWindowTitle("\u26a0")
        else:
            self.setWindowTitle(self.title)
        self.setModal(True)
        self.setStyleSheet('background-color: ' + style.GREY_1 + ';')

    def set_title(self, title: str) -> None:
        """
        This method updates the dialog title.
        Parameters
        ----------
        title : str
            New title to display.
        """

        self.title = title
        self.setWindowTitle(self.title)

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


class HelpDialog(CoCoNetDialog):
    """
    This dialog displays the user guide from the documentation file.
    """

    def __init__(self):
        super().__init__('User Guide', 'User Guide')
        self.resize(QSize(800, 600))
        self.setStyleSheet('background-color: ' + style.GREY_3 + ';')

        # The dialog contains a text area reading the user guide file
        text = open(ROOT_DIR.replace('/coconet', '') +
                    '/docs/coconet/userguide/User Guide.html', encoding="utf8").read()
        text_area = QTextEdit(text)
        text_area.setReadOnly(True)

        self.layout.addWidget(text_area)
        self.render_layout()


class SingleButtonDialog(CoCoNetDialog):
    """
    This class is a generic dialog with a single button
    at the bottom of the main layout. It also provides
    a method for imposing the button text ('Ok' by default)
    Attributes
    ----------
    button : CustomButton
        A single button for leaving the dialog
    Methods
    ----------
    set_button_text(str)
        A method for setting the button text
    """

    def __init__(self, title: str = 'CoCoNet Dialog', message: str = ''):
        super(SingleButtonDialog, self).__init__(title, message)

        self.button = CustomButton('Ok', primary=True)
        self.button.clicked.connect(self.close)

    def set_button_text(self, text: str):
        self.button.setText(text)

    def render_layout(self) -> None:
        """
        Override with a button at the end
        """

        self.layout.addWidget(self.button)
        self.setLayout(self.layout)


class TwoButtonsDialog(CoCoNetDialog):
    """
    This class is a generic dialog with two buttons
    at the bottom of the main layout, for accepting or
    refusing the operations of the dialog. It also provides
    a method for imposing the buttons text ('Cancel' and 'Ok'
    by default)
    Attributes
    ----------
    cancel_btn : CustomButton
        A single button for leaving the dialog without applying
        the changes
    ok_btn : CustomButton
        A single button for leaving the dialog and applying
        the changes
    Methods
    ----------
    set_buttons_text(str, str)
        A method for setting the buttons text
    """

    def __init__(self, title: str = 'CoCoNet Dialog', message: str = ''):
        super(TwoButtonsDialog, self).__init__(title, message)

        self.cancel_btn = CustomButton('Cancel')
        self.cancel_btn.clicked.connect(self.close)

        self.ok_btn = CustomButton('Ok', primary=True)
        self.ok_btn.clicked.connect(self.close)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.cancel_btn)
        self.button_layout.addWidget(self.ok_btn)

    def set_buttons_text(self, cancel_text: str, ok_text: str):
        self.cancel_btn.setText(cancel_text)
        self.ok_btn.setText(ok_text)

    def render_layout(self) -> None:
        """
        Override with a button at the end
        """

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)


class MessageDialog(SingleButtonDialog):
    """
    This class is a Dialog displaying a message to the user.
    """

    def __init__(self, message: str, message_type: MessageType):
        super().__init__('', message)

        # Set the dialog stile depending on message_type
        if message_type == MessageType.MESSAGE:
            title_label = CustomLabel('Message', alignment=Qt.AlignCenter)
            title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        else:
            title_label = CustomLabel('Error', alignment=Qt.AlignCenter)
            title_label.setStyleSheet(style.ERROR_LABEL_STYLE)

        # Set content label
        mess_label = CustomLabel(f"\n{self.content}\n", alignment=Qt.AlignCenter)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)

        self.render_layout()


class ConfirmDialog(TwoButtonsDialog):
    """
    This dialog asks the user the confirmation to clear the
    unsaved workspace before continue.
    Attributes
    ----------
    confirm : bool
        Boolean value to store user decision
    """

    def __init__(self, title, message):
        super().__init__(title, message)

        # Set title label
        title_label = CustomLabel(self.title, primary=True)

        # Set message label
        mess_label = CustomLabel(f"\n{self.content}\n", alignment=Qt.AlignCenter)

        self.confirm = False

        # Add buttons to close the dialog
        self.ok_btn.clicked.connect(self.ok)
        self.cancel_btn.clicked.connect(self.deny)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)

        self.render_layout()

    def ok(self):
        self.confirm = True

    def deny(self):
        self.confirm = False


# Deprecated
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
        title_label = CustomLabel("Input required")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)

        # Set message label
        mess_label = CustomLabel("\n" + self.content + "\n")
        mess_label.setAlignment(Qt.AlignCenter)

        # Set input reading
        self.input = None
        self.input_line = CustomTextBox()
        self.input_line.setValidator(QRegExpValidator(QRegExp(
            ArithmeticValidator.TENSOR.regExp().pattern() + "|" +
            ArithmeticValidator.TENSOR_LIST.regExp().pattern())))

        # Add buttons to close the dialog
        confirm_button = CustomButton("Ok")
        confirm_button.clicked.connect(self.save_input)

        cancel_button = CustomButton("Cancel")
        cancel_button.clicked.connect(self.cancel)

        buttons = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(confirm_button)
        buttons_layout.addWidget(cancel_button)
        buttons.setLayout(buttons_layout)

        # Compose widgets
        self.layout.addWidget(title_label)
        self.layout.addWidget(mess_label)
        self.layout.addWidget(self.input_line)
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
                error_dialog.exec()
        self.close()

    def cancel(self) -> None:
        """
        This method closes the dialog without saving the input read.
        """

        self.input = None
        self.close()


class EditNodeDialog(TwoButtonsDialog):
    """
    This dialog allows to edit the selected node in the canvas.
    Attributes
    ----------
    node: NetworkNode
        Current node to edit, which contains information about parameters to
        display and their types.
    parameters: dict
        Dictionary which connects the name of each parameter to its editable
        field, which can be a CustomTextBox or a CustomComboBox.
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
        super().__init__(node_block.node.name, "")
        g_layout = QGridLayout()
        self.layout.addLayout(g_layout)

        # Connect node
        self.node = node_block
        self.parameters = dict()
        self.edited_data = dict()
        self.has_edits = False

        # Build main_layout
        title_label = CustomLabel("Edit parameters")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        g_layout.addWidget(title_label, 0, 0, 1, 2)

        # Input box
        in_dim_label = CustomLabel("Input")
        in_dim_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        in_dim_label.setAlignment(Qt.AlignRight)
        g_layout.addWidget(in_dim_label, 1, 0)

        in_dim_box = CustomTextBox(','.join(map(str, node_block.in_dim)))
        in_dim_box.setValidator(ArithmeticValidator.TENSOR)

        g_layout.addWidget(in_dim_box, 1, 1)
        self.parameters["in_dim"] = in_dim_box

        if not node_block.is_head:
            in_dim_box.setReadOnly(True)

        # Display parameters if present
        if node_block.node.param:
            self.append_node_params(node_block.node, node_block.block_data, g_layout)

        # "Apply" button which saves changes
        self.set_buttons_text('Discard', 'Apply')
        self.ok_btn.clicked.connect(self.save_data)

        self.render_layout()

    def append_node_params(self, node: NetworkNode, current_data: dict, layout: QGridLayout) -> None:
        """
        This method adds to the dialog layer the editable parameters of
        the node, and returns the last row counter for the grid main_layout.
        Attributes
        ----------
        node : NetworkNode
            The node whose parameters are displayed.
        current_data : dict
            The node current data.
        layout : QGridLayout
            The grid layout to fill
        """

        # Init column counter
        counter = 2

        # Display parameter labels
        for param, value in node.param.items():
            param_label = CustomLabel(param)
            if node.param[param]["editable"] == "false":
                param_label.setStyleSheet(style.UNEDITABLE_PARAM_LABEL_STYLE)

            param_label.setAlignment(Qt.AlignRight)
            # Set the tooltip of the input with the description
            param_label.setToolTip("<" + value["type"] + ">: "
                                   + value["description"])
            layout.addWidget(param_label, counter, 0)

            # Display parameter values
            if value["type"] == "boolean":
                line = CustomComboBox()
                line.addItem("True")
                line.addItem("False")
                line.setPlaceholderText(str(current_data[param]))
            else:
                line = CustomTextBox()
                if node.param[param]["editable"] == "false":
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

            if node.param[param]["editable"] == "false":
                line.setStyleSheet(style.UNEDITABLE_VALUE_LABEL_STYLE)
            layout.addWidget(line, counter, 1)

            # Keep trace of CustomTextBox objects
            self.parameters[param] = line
            counter += 1

    def save_data(self) -> None:
        """
        This method saves the changed parameters in their
        correct format, storing them in a dictionary.
        """

        self.has_edits = True

        for key, line in self.parameters.items():
            try:
                if isinstance(line, CustomTextBox):
                    if line.isModified() and len(line.text()) != 0:
                        if key == "in_dim":
                            self.edited_data["in_dim"] = tuple(
                                map(int, line.text().split(',')))
                        else:
                            data_type = self.node.node.param[key]["type"]

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
                                self.edited_data[key] = u.text_to_tensor_set(line.text())

                elif isinstance(line, CustomComboBox):
                    if line.currentText() == "True":
                        self.edited_data[key] = True
                    else:
                        self.edited_data[key] = False

            except Exception:
                # If there are errors in data format
                error_dialog = MessageDialog("Please check data format.", MessageType.ERROR)
                error_dialog.exec()

        self.close()


class EditSmtPropertyDialog(TwoButtonsDialog):
    """
    This dialog allows to define a generic SMT property
    by writing directly in the SMT-LIB language.
    Attributes
    ----------
    property_block : PropertyBlock
        Current property to edit.
    new_property : str
        New SMT-LIB property string.
    smt_box : CustomTextArea
        Input box.
    has_edits : bool
        Flag signaling if the property was edited.
    Methods
    ----------
    save_data()
        Procedure to return the new property.
    """

    def __init__(self, property_block: PropertyBlock):
        super().__init__("Edit property", "")
        self.property_block = property_block
        self.new_property = self.property_block.smt_string
        self.has_edits = False
        g_layout = QGridLayout()
        self.layout.addLayout(g_layout)

        # Build main_layout
        title_label = CustomLabel("SMT property")
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        g_layout.addWidget(title_label, 0, 0, 1, 2)

        # Input box
        smt_label = CustomLabel("SMT-LIB definition")
        smt_label.setStyleSheet(style.IN_DIM_LABEL_STYLE)
        smt_label.setAlignment(Qt.AlignRight)
        g_layout.addWidget(smt_label, 1, 0)

        self.smt_box = CustomTextArea()
        self.smt_box.insertPlainText(self.new_property)
        g_layout.addWidget(self.smt_box, 1, 1)

        # "Apply" button which saves changes
        self.set_buttons_text('Discard', 'Apply')
        self.ok_btn.clicked.connect(self.save_data)

        self.render_layout()

    def save_data(self):
        self.has_edits = True
        self.new_property = self.smt_box.toPlainText()


class EditLocalRobustnessPropertyDialog(TwoButtonsDialog):
    def __init__(self, property_block: PropertyBlock, nn: SequentialNetwork):
        super().__init__("Edit property", "")
        self.property_block = property_block
        self.has_edits = False
        self.nn = nn

        self.local_input = None
        self.local_output = None
        self.epsilon_noise = 0.0
        self.delta_robustness = 0.0

        grid = QGridLayout()

        # Build main_layout
        title_label = CustomLabel("Local robustness property", primary=True)
        grid.addWidget(title_label, 0, 0, 1, 2)

        # Labels
        in_label = CustomLabel("Local input")
        grid.addWidget(in_label, 1, 0)
        self.local_input_text = CustomTextBox()
        self.local_input_text.setValidator(ArithmeticValidator.SAMPLE)
        grid.addWidget(self.local_input_text, 1, 1)

        out_label = CustomLabel("Local output")
        grid.addWidget(out_label, 2, 0)
        self.local_output_text = CustomTextBox()
        self.local_output_text.setValidator(ArithmeticValidator.SAMPLE)
        grid.addWidget(self.local_output_text, 2, 1)

        eps_label = CustomLabel("Epsilon noise")
        grid.addWidget(eps_label, 3, 0)
        self.epsilon_noise_text = CustomTextBox('0.0')
        self.epsilon_noise_text.setValidator(ArithmeticValidator.FLOAT)
        grid.addWidget(self.epsilon_noise_text, 3, 1)

        delta_label = CustomLabel("Delta robustness")
        grid.addWidget(delta_label, 4, 0)
        self.delta_robustness_text = CustomTextBox('0.0')
        self.delta_robustness_text.setValidator(ArithmeticValidator.FLOAT)
        grid.addWidget(self.delta_robustness_text, 4, 1)

        # Visualizer
        self.viewer = CustomTextArea()
        self.viewer.setReadOnly(True)
        self.viewer.setFixedHeight(80)
        self.viewer.appendPlainText('X <= x0 + eps')
        self.viewer.appendPlainText('-X <= -x0 + eps')
        self.viewer.appendPlainText('')
        self.viewer.appendPlainText('Y <= y0 + delta')
        self.viewer.appendPlainText('-Y <= -y0 + delta')

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.set_buttons_text('Cancel', 'Save')
        self.ok_btn.clicked.connect(self.save_property)

        self.layout.addLayout(grid)
        self.layout.addWidget(self.viewer, 2)

        self.render_layout()

    def save_property(self):
        self.local_input = self.local_input_text.text().split(',')
        self.local_output = self.local_output_text.text().split(',')
        self.epsilon_noise = self.epsilon_noise_text.text()
        self.delta_robustness = self.delta_robustness_text.text()

        if len(self.local_input) == self.nn.get_input_len() and len(self.local_output) == self.nn.get_output_len():
            self.has_edits = True

        self.close()


class EditPolyhedralPropertyDialog(CoCoNetDialog):
    """
    This dialog allows to define a polyhedral property
    within a controlled environment.
    Attributes
    ----------
    property_block : PropertyBlock
        Current property to edit.
    property_list : list
        List of given properties.
    has_edits : bool
        Flag signaling if the property was edited.
    Methods
    ----------
    add_entry(str, str, str)
        Procedure to append the current constraint to the property list.
    save_property()
        Procedure to return the defined property.
    """

    def __init__(self, property_block: PropertyBlock):
        super().__init__("Edit property", "")
        self.property_block = property_block
        self.has_edits = False
        self.property_list = []
        self.viewer = CustomTextArea()
        self.viewer.setReadOnly(True)
        self.viewer.setFixedHeight(100)
        grid = QGridLayout()

        # Build main_layout
        title_label = CustomLabel("Polyhedral property", primary=True)
        grid.addWidget(title_label, 0, 0, 1, 3)

        # Labels
        var_label = CustomLabel("Variable", primary=True)
        grid.addWidget(var_label, 1, 0)

        relop_label = CustomLabel("Operator", primary=True)
        grid.addWidget(relop_label, 1, 1)

        value_label = CustomLabel("Value", primary=True)
        grid.addWidget(value_label, 1, 2)

        self.var_cb = CustomComboBox()
        for v in property_block.variables:
            self.var_cb.addItem(v)
        grid.addWidget(self.var_cb, 2, 0)

        self.op_cb = CustomComboBox()
        operators = ["<=", "<", ">", ">="]
        for o in operators:
            self.op_cb.addItem(o)
        grid.addWidget(self.op_cb, 2, 1)

        self.val = CustomTextBox()
        self.val.setValidator(ArithmeticValidator.FLOAT)
        grid.addWidget(self.val, 2, 2)

        # "Add" button which adds the constraint
        add_button = CustomButton("Add")
        add_button.clicked.connect(
            lambda: self.add_entry(self.var_cb.currentText(), self.op_cb.currentText(), self.val.text()))
        grid.addWidget(add_button, 3, 0)

        # "Cancel" button which closes the dialog without saving
        cancel_button = CustomButton("Cancel")
        cancel_button.clicked.connect(self.close)
        grid.addWidget(cancel_button, 3, 1)

        # "Save" button which saves the state
        save_button = CustomButton("Save", primary=True)
        save_button.clicked.connect(self.save_property)
        grid.addWidget(save_button, 3, 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        self.layout.addLayout(grid)
        self.layout.addWidget(self.viewer, 3)
        self.render_layout()

    def add_entry(self, var: str, op: str, val: str) -> None:
        self.property_list.append((var, op, val))
        self.viewer.appendPlainText(f"{var} {op} {val}")
        self.var_cb.setCurrentIndex(0)
        self.op_cb.setCurrentIndex(0)
        self.val.setText("")

    def save_property(self) -> None:
        self.has_edits = True
        if self.val.text() != '':
            self.add_entry(str(self.var_cb.currentText()),
                           str(self.op_cb.currentText()),
                           self.val.text())
        self.close()
