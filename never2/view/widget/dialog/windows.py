import json
import logging
from typing import Callable

import pynever.datasets as dt
import torch
import torch.nn.functional as fun
import torch.optim as opt
import torchvision.transforms as tr
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QPushButton, \
    QFileDialog
from pynever.datasets import Dataset
from pynever.networks import NeuralNetwork
from pynever.strategies.training import PytorchTraining, PytorchMetrics

import never2.view.styles as style
import never2.view.util.utility as u
from never2.view.widget.dialog.dialogs import MessageDialog, MessageType, GenericDatasetDialog, ArithmeticValidator
from never2.view.widget.misc import LoggerTextBox


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
    create_widget_layout(str, dict, Callable, Callable)
        Procedure to display widgets from a dictionary.

    """

    def __init__(self, title="NeVer Window", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.title = title
        self.widgets = dict()

        self.setWindowTitle(self.title)
        self.setModal(True)
        self.setStyleSheet("background-color: " + style.GREY_1 + ";")

    def render_layout(self) -> None:
        """
        This method updates the main_layout with the changes done
        in the child class(es).

        """

        self.setLayout(self.layout)

    def create_widget_layout(self, layout_name: str, widget_dict: dict,
                             cb_f: Callable = None, line_f: Callable = None) -> QHBoxLayout:
        """
        This method sets up the parameters layout by reading
        the JSON-based dict of train_params and building
        the corresponding graphic objects.

        Parameters
        ----------
        layout_name : str
            The name of the layout to be shown in the label.
        widget_dict : dict
            The dictionary of widgets to build.
        cb_f : Callable, optional
            The activation function for combo boxes.
        line_f : Callable, optional
            The activation function for text boxes.

        Returns
        ----------
        QHBoxLayout
            The layout with all the widgets loaded.

        """

        widget_layout = QHBoxLayout()
        left_layout = QGridLayout()
        left_layout.setAlignment(Qt.AlignTop)

        title = QLabel(layout_name)
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title, 0, 0, 1, 2)

        counter = 1
        for first_level in widget_dict.keys():
            sub_key = next(iter(widget_dict[first_level]))
            if type(widget_dict[first_level][sub_key]) == dict:
                self.widgets[first_level] = QComboBox()
                for second_level in widget_dict[first_level].keys():
                    self.widgets[first_level].addItem(second_level)
                self.widgets[first_level].setCurrentIndex(-1)
                self.widgets[first_level].activated.connect(cb_f(first_level))
            else:
                if widget_dict[first_level]["type"] == "bool":
                    self.widgets[first_level] = QComboBox()
                    self.widgets[first_level].addItems([str(widget_dict[first_level]["value"]),
                                                        str(not widget_dict[first_level]["value"])])
                else:
                    self.widgets[first_level] = QLineEdit()
                    self.widgets[first_level].setText(str(widget_dict[first_level].get("value", "")))
                    self.widgets[first_level].textChanged.connect(line_f(first_level))
                    if widget_dict[first_level]["type"] == "int":
                        self.widgets[first_level].setValidator(ArithmeticValidator.INT)
                    elif widget_dict[first_level]["type"] == "float":
                        self.widgets[first_level].setValidator(ArithmeticValidator.FLOAT)
                    elif widget_dict[first_level]["type"] == "tensor" or \
                            widget_dict[first_level]["type"] == "tuple":
                        self.widgets[first_level].setValidator(ArithmeticValidator.TENSOR)

            w_label = QLabel(first_level)
            w_label.setToolTip(widget_dict[first_level].get("description"))
            left_layout.addWidget(w_label, counter, 0)
            left_layout.addWidget(self.widgets[first_level], counter, 1)
            counter += 1

        widget_layout.addLayout(left_layout)
        return widget_layout


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
    dataset_path : str
        The dataset path to train the network.
    dataset_params : dict
        Additional parameters for generic datasets.
    dataset_transform : Transform
        Transform on the dataset.
    train_params : dict
        The parameters required by pyNeVer to correctly
        train the network.
    gui_params : dict
        The dictionary of secondary parameters displayed
        based on the selection.
    grid_layout : QGridLayout
        The layout to display the GUI parameters on.

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
        self.dataset_params = dict()
        self.dataset_transform = None
        self.train_params = dict()
        self.gui_params = dict()
        self.grid_layout = QGridLayout()

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
        self.widgets["dataset"].activated \
            .connect(lambda: self.setup_dataset(self.widgets["dataset"].currentText()))
        dataset_layout.addWidget(QLabel("Dataset"))
        dataset_layout.addWidget(self.widgets["dataset"])
        self.layout.addLayout(dataset_layout)

        transform_layout = QHBoxLayout()
        self.widgets["transform"] = QComboBox()
        self.widgets["transform"] \
            .addItem("Compose(ToTensor, Normalize(1, 0.5), Flatten)")
        self.widgets["transform"].setCurrentIndex(-1)
        self.widgets["transform"].activated \
            .connect(lambda: self.setup_transform(self.widgets["transform"].currentText()))
        transform_layout.addWidget(QLabel("Dataset transform"))
        transform_layout.addWidget(self.widgets["transform"])
        self.layout.addLayout(transform_layout)

        # Separator
        sep_label = QLabel("Training parameters")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet(style.NODE_LABEL_STYLE)
        self.layout.addWidget(sep_label)

        # Main body
        # Activation functions for dynamic widgets
        def activation_combo(key: str):
            return lambda: self.update_grid_view(f"{key}:{self.widgets[key].currentText()}")

        def activation_line(key: str):
            return lambda: self.update_dict_value(key, "", self.widgets[key].text())

        body_layout = self.create_widget_layout("", self.train_params,
                                                activation_combo, activation_line)
        body_layout.addLayout(self.grid_layout)
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.layout.addLayout(body_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train network")
        self.train_btn.clicked.connect(self.train_network)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.cancel_btn)
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
            def activation_combo(super_key: str, key: str):
                return lambda: self.update_dict_value(name,
                                                      key,
                                                      widgets_2level[f"{super_key}:{key}"][1].currentText())

            def activation_line(super_key: str, key: str):
                return lambda: self.update_dict_value(name,
                                                      key,
                                                      widgets_2level[f"{super_key}:{key}"][1].text())

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
                if v["type"] == "int":
                    widgets_2level[f"{name}:{k}"][1].setValidator(ArithmeticValidator.INT)
                elif v["type"] == "float":
                    widgets_2level[f"{name}:{k}"][1].setValidator(ArithmeticValidator.FLOAT)
                elif v["type"] == "tensor" or \
                        v["type"] == "tuple":
                    widgets_2level[f"{name}:{k}"][1].setValidator(ArithmeticValidator.TENSOR)

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
        elif gui_param["type"] == "int" and value != "":
            value = int(value)
        elif gui_param["type"] == "float" and value != "":
            value = float(value)
        elif gui_param["type"] == "tuple" and value != "":
            value = eval(value)

        # Apply changes
        if ":" in name:
            first_level, second_level = name.split(":")
            self.train_params[first_level][second_level][key]["value"] = value
        else:
            self.train_params[name]["value"] = value

    def setup_dataset(self, name: str) -> None:
        """
        This method reacts to the selection of a dataset in the
        dataset combo box. Depending on the selection, the correct
        path is saved and any additional parameters are asked.

        Parameters
        ----------
        name : str
            The dataset name.

        """

        if name == "MNIST":
            self.dataset_path = "data/MNIST/"
        elif name == "Fashion MNIST":
            self.dataset_path = "data/fMNIST/"
        else:
            datapath = QFileDialog.getOpenFileName(None, "Select data source...", "")
            self.dataset_path = datapath[0]

            # Get additional parameters via dialog
            dialog = GenericDatasetDialog()
            dialog.exec()
            self.dataset_params = dialog.params

    def setup_transform(self, sel_t: str) -> None:
        if sel_t != '':
            self.dataset_transform = tr.Compose([tr.ToTensor(),
                                                 tr.Normalize(1, 0.5),
                                                 tr.Lambda(lambda x: torch.flatten(x))])

    def load_dataset(self) -> Dataset:
        """
        This method initializes the selected dataset object,
        given the path loaded before.

        Returns
        -------
        Dataset
            The dataset object.

        """
        if self.dataset_path == "data/MNIST/":
            return dt.TorchMNIST(self.dataset_path, True, self.dataset_transform)
        elif self.dataset_path == "data/fMNIST/":
            return dt.TorchFMNIST(self.dataset_path, True, self.dataset_transform)
        elif self.dataset_path != "":
            return dt.GenericFileDataset(self.dataset_path,
                                         self.dataset_params["target_idx"],
                                         self.dataset_params["data_type"],
                                         self.dataset_params["delimiter"],
                                         self.dataset_transform)

    def train_network(self):
        # TODO MOVE CHECKS?
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

        # Add logger text box
        log_textbox = LoggerTextBox(self)
        logger = logging.getLogger("pynever.strategies.training")
        logger.addHandler(log_textbox)
        logger.setLevel(logging.INFO)
        self.layout.addWidget(log_textbox.widget)

        # Load dataset
        logger.info("***** NeVer 2 - GET DATASET *****")
        logger.info("Collecting data...")
        data = self.load_dataset()
        logger.info("Done.\n\n")

        logger.info("***** NeVer 2 - TRAINING *****")

        # Create optimizer dictionary of parameters
        opt_params = dict()
        for k, v in self.gui_params["Optimizer:Adam"].items():
            opt_params[v["name"]] = v["value"]

        # Create scheduler dictionary of parameters
        sched_params = dict()
        for k, v in self.gui_params["Scheduler:ReduceLROnPlateau"].items():
            sched_params[v["name"]] = v["value"]

        # Init train strategy
        train = PytorchTraining(opt.Adam, opt_params,
                                fun.cross_entropy,
                                self.train_params["Epochs"]["value"],
                                self.train_params["Validation percentage"]["value"],
                                self.train_params["Training batch size"]["value"],
                                self.train_params["Validation batch size"]["value"],
                                opt.lr_scheduler.ReduceLROnPlateau,
                                sched_params,
                                PytorchMetrics.inaccuracy)
        train.train(self.nn, data)
        self.train_btn.setEnabled(False)
        self.cancel_btn.setText("Close")


class VerificationWindow(NeVerWindow):
    def __init__(self):
        super().__init__("Verify network")
