import json

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QPushButton

import never2.view.styles as style
from never2.core.controller.pynevertemp.networks import NeuralNetwork
from never2.view.util.utility import allow_list_in_dict


class NeVerWindow(QtWidgets.QDialog):
    """
    Base class for grouping common elements of the windows.
    Each window shares a main layout (vertical by default),
    a title and a dictionary of combobox for the parameters.

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


class TrainingWindow(NeVerWindow):
    """
    This class is a Window for the training of the network.
    It features a file picker for choosing the dataset and
    a grid of parameters for tuning the procedure.

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, nn: NeuralNetwork):
        super().__init__("Train Network")

        # Training elements
        self.nn = nn
        self.widgets = dict()
        self.train_params = dict()
        with open('never2/res/json/training.json') as json_file:
            # Init dict with default values
            self.train_params = json.loads(json_file.read())
            self.train_params = allow_list_in_dict(self.train_params)

        # Dataset
        dataset_layout = QHBoxLayout()
        dataset_picker = QComboBox()
        dataset_picker.addItems(["MNIST", "Fashion MNIST", "James", "..."])
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
            if type(self.train_params[first_level]) == dict:
                self.widgets[first_level] = QComboBox()
                for second_level in self.train_params[first_level].keys():
                    self.widgets[first_level].addItem(second_level)
                self.widgets[first_level].setCurrentIndex(-1)
            else:
                self.widgets[str(first_level)] = QLineEdit()

            params_layout.addWidget(QLabel(first_level), counter, 0)
            params_layout.addWidget(self.widgets[first_level], counter, 1)
            counter += 1

        self.widgets["Optimizer"].activated.connect(
            lambda: self.details_layout.update_view("Optimizer:" + self.widgets["Optimizer"].currentText()))
        self.widgets["Scheduler"].activated.connect(
            lambda: self.details_layout.update_view("Scheduler:" + self.widgets["Scheduler"].currentText()))
        self.widgets["Loss Function"].activated.connect(
            lambda: self.details_layout.update_view("Loss Function:" + self.widgets["Loss Function"].currentText()))
        self.widgets["Metrics"].activated.connect(
            lambda: self.details_layout.update_view("Metrics:" + self.widgets["Metrics"].currentText()))

        body_layout.addLayout(params_layout)

        self.details_layout = GUIParamLayout(self.train_params)
        self.details_layout.setAlignment(Qt.AlignTop)
        body_layout.addLayout(self.details_layout)
        self.layout.addLayout(body_layout)

        # Separator
        sep_label = QLabel("***")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet(style.NODE_LABEL_STYLE)
        self.layout.addWidget(sep_label)

        # Buttons
        btn_layout = QHBoxLayout()
        train_btn = QPushButton("Train network")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(train_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

        self.render_layout()

    # def train_dataset(self, dataset: Dataset):
    #     train = PytorchTraining(self.optimizer, self.details_layout.params["Optimizer:Adam"],
    #                             self.scheduler, self.details_layout.params["Scheduler:ReduceLROnPlateau"],
    #                             self.loss_function, self.details_layout.params["Loss Function:MSE Loss"],
    #                             self.metrics, self.details_layout.params["Metrics:MSE Loss"],
    #                             10, 30, 5, 5)
    #     train.train(self.nn, dataset)


class GUIParamLayout(QVBoxLayout):
    """
    This class is a layout for showing the possible parameters
    of the selected training element. It features a grid with
    pairs <QLabel, QWidget> for reading parameters.

    Attributes
    ----------
    params : dict
        Dictionary of training parameters.
        Structured as: {<training_par>: <gui_params>}
        where <gui_params> = {<param>: <default_value>}.
    grid_dict : dict
        Dictionary of graphical pairs (QLabel, QWidget)
        for displaying and editing the training parameters.
    grid_layout : QGridLayout
        Layout containing the second-level parameters.

    Methods
    ----------
    clear_grid()
        Procedure to clear the grid layout.
    update_view(str)
        Procedure to update the grid layout.
    show_layout(str)
        Procedure to display the grid layout.
    update_dict_value(str, str, str)
        Procedure to update the parameters.

    """

    def __init__(self, params: dict):
        super().__init__()

        self.all_params = params
        self.gui_params = dict()
        self.grid_dict = dict()
        self.grid_layout = QGridLayout()
        self.addLayout(self.grid_layout)

    def clear_grid(self) -> None:
        """
        This method clears the grid view of the layout,
        in order to display fresh new infos.

        """

        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

    def update_view(self, caller: str) -> None:
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

        for first_level in self.all_params.keys():
            if type(self.all_params[first_level]) == dict:
                for second_level in self.all_params[first_level].keys():
                    if caller == f"{first_level}:{second_level}" and caller not in self.gui_params:
                        self.gui_params[caller] = self.all_params[first_level][second_level]

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

        count = 1
        for k, v in self.gui_params[name].items():
            if type(v) == list:
                cb = QComboBox()
                cb.addItems(v)
                self.grid_dict[k] = (QLabel(k), cb)
                self.grid_dict[k][1].activated.connect(
                    lambda: self.update_dict_value(name, k, self.grid_dict[k][1].currentText()))
            else:
                self.grid_dict[k] = (QLabel(k), QLineEdit(str(v)))
                self.grid_dict[k][1].textChanged.connect(
                    lambda: self.update_dict_value(name, k, self.grid_dict[k][1].text()))

            self.grid_layout.addWidget(self.grid_dict[k][0], count, 0)
            self.grid_layout.addWidget(self.grid_dict[k][1], count, 1)
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
            The new value for parameter[name][k].

        """

        param = self.gui_params[name][key]

        # Update list putting the selected value at index 0
        if type(param) == list:
            param.remove(value)
            self.gui_params[name][key] = [value] + param
        else:
            self.gui_params[name][key] = value
