import json

import torch.nn.functional as fun
import torch.optim as opt
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QPushButton, \
    QFileDialog

import never2.view.styles as style
import never2.view.util.utility as u
from never2.core.controller.pynevertemp.networks import NeuralNetwork
from never2.core.controller.pynevertemp.strategies.training import PytorchTraining, PytorchMetrics
from never2.view.widget.dialog.dialogs import MessageDialog, MessageType


class NeVerWindow(QtWidgets.QDialog):
    """
    Base class for grouping common elements of the windows.
    Each window shares a main layout (vertical by default),
    a title and a dictionary of combobox for the parameters.

    Attributes
    ----------
    layout : QVBoxLayout
        The main layout of the window.
    title : str
        Window title to display.
    widgets : dict
        The dictionary of the displayed widgets.

    Methods
    ----------
    render_layout()
        Procedure to display the window layout.

    """

    def __init__(self, title="NeVer Window", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.title = title
        self.widgets = dict()
        self.gui_params = dict()
        self.grid_layout = QGridLayout()

        self.setWindowTitle(self.title)
        self.setModal(True)
        self.setStyleSheet("background-color: " + style.GREY_1 + ";")

    def render_layout(self) -> None:
        """
        This method updates the main_layout with the changes done
        in the child class(es).

        """

        self.setLayout(self.layout)


class TrainingWindow(NeVerWindow):
    """
    This class is a Window for the training of the network.
    It features a file picker for choosing the dataset and
    a grid of parameters for tuning the procedure.

    Attributes
    ----------
    nn : NeuralNetwork
        The current network used in the main window, to be
        trained with the parameters selected here.
    dataset : Dataset
        The dataset on which the network is trained.
    train_params : dict
        The parameters required by pyNeVer to correctly
        train the network.

    Methods
    ----------
    clear_grid()
        Procedure to clear the grid layout.
    update_grid_view(str)
        Procedure to update the grid layout.
    show_layout(str)
        Procedure to display the grid layout.
    update_dict_value(str, str, str)
        Procedure to update the parameters.

    """

    def __init__(self, nn: NeuralNetwork):
        super().__init__("Train Network")

        # Training elements
        self.nn = nn
        self.dataset_path = ""
        self.train_params = dict()
        with open('never2/res/json/training.json') as json_file:
            # Init dict with default values
            self.train_params = json.loads(json_file.read())
            # Update dict with types
            self.train_params = u.allow_list_in_dict(self.train_params)
            self.train_params = u.force_types(self.train_params)

        # Dataset
        dataset_layout = QHBoxLayout()
        self.widgets["dataset"] = QComboBox()
        self.widgets["dataset"].addItems(["MNIST", "Fashion MNIST", "Custom data source..."])
        self.widgets["dataset"].setCurrentIndex(-1)
        self.widgets["dataset"].activated.connect(lambda: self.setup_dataset(self.widgets["dataset"].currentText()))
        dataset_layout.addWidget(QLabel("Dataset"))
        dataset_layout.addWidget(self.widgets["dataset"])
        self.layout.addLayout(dataset_layout)

        # Separator
        sep_label = QLabel("***")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet(style.NODE_LABEL_STYLE)
        self.layout.addWidget(sep_label)

        # Main body
        body_layout = QHBoxLayout()
        params_layout = QGridLayout()
        params_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Training parameters")
        title.setAlignment(Qt.AlignCenter)
        params_layout.addWidget(title, 0, 0, 1, 2)

        # Widgets builder
        counter = 1
        for first_level in self.train_params.keys():
            # Activation functions for dynamic widgets
            def activation_combo(key: str):
                return lambda: self.update_grid_view(f"{key}:{self.widgets[key].currentText()}")

            def activation_line(key: str):
                return lambda: self.update_dict_value(key, "", self.widgets[key].text())

            sub_key = next(iter(self.train_params[first_level]))
            if type(self.train_params[first_level][sub_key]) == dict:
                self.widgets[first_level] = QComboBox()
                for second_level in self.train_params[first_level].keys():
                    self.widgets[first_level].addItem(second_level)
                self.widgets[first_level].setCurrentIndex(-1)
                self.widgets[first_level].activated.connect(activation_combo(first_level))
            else:
                self.widgets[first_level] = QLineEdit()
                self.widgets[first_level].setText(str(self.train_params[first_level].get("value", "")))
                self.widgets[first_level].textChanged.connect(activation_line(first_level))

            w_label = QLabel(first_level)
            w_label.setToolTip(self.train_params[first_level].get("description"))
            params_layout.addWidget(w_label, counter, 0)
            params_layout.addWidget(self.widgets[first_level], counter, 1)
            counter += 1

        body_layout.addLayout(params_layout)
        body_layout.addLayout(self.grid_layout)
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.layout.addLayout(body_layout)

        # Separator
        sep_label = QLabel("***")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet(style.NODE_LABEL_STYLE)
        self.layout.addWidget(sep_label)

        # Buttons
        btn_layout = QHBoxLayout()
        train_btn = QPushButton("Train network")
        train_btn.clicked.connect(self.train_network)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(train_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

        self.render_layout()

    def clear_grid(self) -> None:
        """
        This method clears the grid view of the layout,
        in order to display fresh new infos.

        """

        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

    def update_grid_view(self, caller: str) -> None:
        """
        This method updates the grid view of the layout,
        displaying the corresponding parameters to the
        selected parameter.

        Parameters
        ----------
        caller : str
            The parameter selected in the combo box.

        """

        self.clear_grid()

        for first_level in self.train_params.keys():
            if type(self.train_params[first_level]) == dict:
                for second_level in self.train_params[first_level].keys():
                    if caller == f"{first_level}:{second_level}" and caller not in self.gui_params:
                        self.gui_params[caller] = self.train_params[first_level][second_level]

        self.show_layout(caller)

    def show_layout(self, name: str) -> None:
        """
        This method displays a grid layout initialized by the
        dictionary of parameters and default values.

        Parameters
        ----------
        name : str
            The name of the main parameter to which
            the dictionary is related.

        """

        title = QLabel(name)
        title.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(title, 0, 0, 1, 2)
        widgets_2level = dict()

        count = 1
        for k, v in self.gui_params[name].items():
            # Activation functions for dynamic widgets
            def activation_combo(superkey: str, key: str):
                return lambda: self.update_dict_value(name,
                                                      key,
                                                      widgets_2level[f"{superkey}-{key}"][1].currentText())

            def activation_line(superkey: str, key: str):
                return lambda: self.update_dict_value(name,
                                                      key,
                                                      widgets_2level[f"{superkey}-{key}"][1].text())

            w_label = QLabel(k)
            w_label.setToolTip(v.get("description"))
            if v["type"] == "bool":
                cb = QComboBox()
                cb.addItems([str(v["value"]), str(not v["value"])])
                widgets_2level[f"{name}:{k}"] = (w_label, cb)
                widgets_2level[f"{name}:{k}"][1].activated.connect(activation_combo(name, k))
            elif "allowed" in v.keys():
                cb = QComboBox()
                cb.addItems(v["allowed"])
                widgets_2level[f"{name}:{k}"] = (w_label, cb)
                widgets_2level[f"{name}:{k}"][1].activated.connect(activation_combo(name, k))
            else:
                widgets_2level[f"{name}:{k}"] = (w_label, QLineEdit(str(v["value"])))
                widgets_2level[f"{name}:{k}"][1].textChanged.connect(activation_line(name, k))

            self.grid_layout.addWidget(widgets_2level[f"{name}:{k}"][0], count, 0)
            self.grid_layout.addWidget(widgets_2level[f"{name}:{k}"][1], count, 1)
            count += 1

    def update_dict_value(self, name: str, key: str, value: str) -> None:
        """
        This method updates the correct parameter based
        on the selection in the GUI. It provides the details
        to access the parameter and the new value to register.

        Parameters
        ----------
        name : str
            The learning parameter name, which is
            the key of the main dict.
        key : str
            The name of the parameter detail,
            which is the key of the second-level dict.
        value : str
            The new value for parameter[name][key].

        """

        # Cast type
        if name not in self.gui_params.keys():
            gui_param = self.train_params[name]
        else:
            gui_param = self.gui_params[name][key]

        if gui_param["type"] == "bool":
            value = value == "True"
        elif gui_param["type"] == "int":
            value = int(value)
        elif gui_param["type"] == "float":
            value = float(value)
        elif gui_param["type"] == "tuple":
            value = eval(value)

        # Apply changes
        if ":" in name:
            first_level, second_level = name.split(":")
            self.train_params[first_level][second_level][key]["value"] = value
        else:
            self.train_params[name]["value"] = value

    def setup_dataset(self, name: str):
        if name == "MNIST":
            self.dataset_path = "data/MNIST/"
        elif name == "Fashion MNIST":
            self.dataset_path = "data/fMNIST/"
        else:
            datapath = QFileDialog.getOpenFileName(None, "Select data source...", "")
            self.dataset_path = datapath[0]

    def train_network(self):
        err_dialog = None
        if self.dataset_path == "":
            err_dialog = MessageDialog("No dataset selected.", MessageType.ERROR)
        elif self.widgets["Optimizer"].currentIndex() == -1:
            err_dialog = MessageDialog("No optimizer selected.", MessageType.ERROR)
        elif self.widgets["Scheduler"].currentIndex() == -1:
            err_dialog = MessageDialog("No scheduler selected.", MessageType.ERROR)
        elif self.widgets["Loss Function"].currentIndex() == -1:
            err_dialog = MessageDialog("No loss function selected.", MessageType.ERROR)
        elif self.widgets["Metrics"].currentIndex() == -1:
            err_dialog = MessageDialog("No metrics selected.", MessageType.ERROR)
        elif "value" not in self.train_params["Epochs"].keys():
            err_dialog = MessageDialog("No epochs selected.", MessageType.ERROR)
        elif "value" not in self.train_params["Validation percentage"].keys():
            err_dialog = MessageDialog("No validation percentage selected.", MessageType.ERROR)
        elif "value" not in self.train_params["Training batch size"].keys():
            err_dialog = MessageDialog("No training batch size selected.", MessageType.ERROR)
        elif "value" not in self.train_params["Validation batch size"].keys():
            err_dialog = MessageDialog("No validation batch size selected.", MessageType.ERROR)
        if err_dialog is not None:
            err_dialog.show()
            return

        train = PytorchTraining(opt.Adam, self.gui_params["Optimizer:Adam"],
                                fun.cross_entropy,
                                3, 0.2, 512, 64,
                                opt.lr_scheduler.ReduceLROnPlateau,
                                self.gui_params["Scheduler:ReduceLROnPlateau"],
                                PytorchMetrics.inaccuracy)
