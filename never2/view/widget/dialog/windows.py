import json

import torch.optim
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QPushButton
from torch.nn.functional import cross_entropy

import never2.core.controller.pynevertemp.datasets as dt
import never2.view.styles as style
import never2.view.util.utility as u
from never2.core.controller.pynevertemp.networks import NeuralNetwork
from never2.core.controller.pynevertemp.strategies.training import PytorchTraining, PytorchMetrics


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
        self.dataset = None
        self.train_params = dict()
        with open('never2/res/json/training.json') as json_file:
            # Init dict with default values
            self.train_params = json.loads(json_file.read())
            # Update dict with types
            self.train_params = u.allow_list_in_dict(self.train_params)
            self.train_params = u.force_types(self.train_params)

        # Dataset
        dataset_layout = QHBoxLayout()
        dataset_picker = QComboBox()
        dataset_picker.addItems(["MNIST", "Fashion MNIST", "James", "..."])
        dataset_picker.setCurrentIndex(-1)
        dataset_picker.activated.connect(lambda: self.load_dataset(dataset_picker.currentText()))
        dataset_layout.addWidget(QLabel("Dataset"))
        dataset_layout.addWidget(dataset_picker)
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
            subkey = next(iter(self.train_params[first_level]))
            if type(self.train_params[first_level][subkey]) == dict:
                self.widgets[first_level] = QComboBox()
                for second_level in self.train_params[first_level].keys():
                    self.widgets[first_level].addItem(second_level)
                self.widgets[first_level].setCurrentIndex(-1)
            else:
                self.widgets[first_level] = QLineEdit()
                self.widgets[first_level].setText(str(self.train_params[first_level]["value"]))

            params_layout.addWidget(QLabel(first_level), counter, 0)
            params_layout.addWidget(self.widgets[first_level], counter, 1)
            counter += 1

        self.widgets["Optimizer"].activated.connect(
            lambda: self.update_grid_view("Optimizer:" + self.widgets["Optimizer"].currentText()))
        self.widgets["Scheduler"].activated.connect(
            lambda: self.update_grid_view("Scheduler:" + self.widgets["Scheduler"].currentText()))
        self.widgets["Loss Function"].activated.connect(
            lambda: self.update_grid_view("Loss Function:" + self.widgets["Loss Function"].currentText()))
        self.widgets["Metrics"].activated.connect(
            lambda: self.update_grid_view("Metrics:" + self.widgets["Metrics"].currentText()))
        self.widgets["Epochs"].textChanged.connect(
            lambda: self.update_dict_value("Epochs", "", self.widgets["Epochs"].text()))

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

    def load_dataset(self, name: str):
        # TODO DISPLAY LOADING
        if name == "MNIST":
            self.dataset = dt.TorchMNIST("data/", True)
        elif name == "Fashion MNIST":
            self.dataset = dt.TorchFMNIST("data/", True)
        elif name == "James":
            self.dataset = dt.DynamicsJamesPos("data/", True)
        else:
            self.dataset = dt.GenericFileDataset("data/", 0)

    def train_network(self):
        train = PytorchTraining(torch.optim.Adam, self.gui_params["Optimizer:Adam"],
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                                self.gui_params["Scheduler:ReduceLROnPlateau"],
                                cross_entropy,
                                self.gui_params["Loss Function:Cross Entropy"],
                                PytorchMetrics.inaccuracy, self.gui_params["Metrics:Inaccuracy"],
                                3, 0.2, 512, 64)
        train.train(self.nn, self.dataset)

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
        grid_dict = dict()

        count = 1
        for k, v in self.gui_params[name].items():
            long_k = f"{name}:{k}"
            if v["type"] == "bool":
                cb = QComboBox()
                cb.addItems([str(v["value"]), str(not v["value"])])
                grid_dict[long_k] = (QLabel(k), cb)
            elif "allowed" in v.keys():
                cb = QComboBox()
                cb.addItems(v["allowed"])
                grid_dict[long_k] = (QLabel(k), cb)
            else:
                grid_dict[long_k] = (QLabel(k), QLineEdit(str(v["value"])))

            self.grid_layout.addWidget(grid_dict[long_k][0], count, 0)
            self.grid_layout.addWidget(grid_dict[long_k][1], count, 1)
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
