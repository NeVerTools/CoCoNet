from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QPushButton

import never2.view.styles as style


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
        self.params = dict()

        self.setWindowTitle("\u26a0")
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

    def __init__(self):
        super().__init__("Train Network")

        # Title
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(style.NODE_LABEL_STYLE)
        self.layout.addWidget(title_label)

        # Dataset
        dataset_layout = QHBoxLayout()
        dataset_picker = QComboBox()
        dataset_picker.addItems(["James", "MNIST", "fMNIST", "..."])
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

        self.params["Optimizer"] = QComboBox()
        self.params["Optimizer"].addItem("Adam")
        self.params["Optimizer"].setCurrentIndex(-1)
        self.params["Optimizer"].activated.connect(
            lambda: self.details_layout.update_view("Optimizer:" + self.params["Optimizer"].currentText()))

        self.params["Scheduler"] = QComboBox()
        self.params["Scheduler"].addItem("ReduceLROnPlateau")
        self.params["Scheduler"].setCurrentIndex(-1)
        self.params["Scheduler"].activated.connect(
            lambda: self.details_layout.update_view("Scheduler:" + self.params["Scheduler"].currentText()))

        self.params["Loss Function"] = QComboBox()
        self.params["Loss Function"].addItems(["Cross Entropy",
                                               "MSE Loss"])
        self.params["Loss Function"].setCurrentIndex(-1)
        self.params["Loss Function"].activated.connect(
            lambda: self.details_layout.update_view("Loss Function:" + self.params["Loss Function"].currentText()))

        self.params["Metrics"] = QComboBox()
        self.params["Metrics"].addItems(["Inaccuracy",
                                         "MSE Loss"])
        self.params["Metrics"].setCurrentIndex(-1)
        self.params["Metrics"].activated.connect(
            lambda: self.details_layout.update_view("Metrics:" + self.params["Metrics"].currentText()))

        params_layout.addWidget(QLabel("Optimizer:"), 1, 0)
        params_layout.addWidget(self.params["Optimizer"], 1, 1)
        params_layout.addWidget(QLabel("Scheduler:"), 2, 0)
        params_layout.addWidget(self.params["Scheduler"], 2, 1)
        params_layout.addWidget(QLabel("Loss Function:"), 3, 0)
        params_layout.addWidget(self.params["Loss Function"], 3, 1)
        params_layout.addWidget(QLabel("Metrics:"), 4, 0)
        params_layout.addWidget(self.params["Metrics"], 4, 1)
        body_layout.addLayout(params_layout)

        self.details_layout = GUIParamLayout()
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


class GUIParamLayout(QVBoxLayout):
    """
    This class is a layout for showing the possible parameters
    of the selected training element. It features a grid with
    pairs <label, QLineEdit> for reading parameters.

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self):
        super().__init__()

        self.grid_dict = dict()
        self.grid_layout = QGridLayout()

        # Default init
        self.params = dict()
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

        if caller == "Optimizer:Adam" and caller not in self.params:
            self.params[caller] = {"Learning rate": 1e-3,
                                   "Betas": (0.9, 0.999),
                                   "Epsilon": 1e-8,
                                   "Weight decay": 0,
                                   "AMSGrad": ["False", "True"]}
        elif caller == "Scheduler:ReduceLROnPlateau" and caller not in self.params:
            self.params[caller] = {"Mode": ["min", "max"],
                                   "Factor": 0.1,
                                   "Patience": 10,
                                   "Threshold": 1e-4,
                                   "Threshold_mode": ["rel", "abs"],
                                   "Cooldown": 0,
                                   "Min LR": 0,
                                   "Eps": 1e-8,
                                   "Verbose": ["False", "True"]}
        elif caller == "Loss Function:Cross Entropy" and caller not in self.params:
            self.params[caller] = {"Weight": (),
                                   "Size average": ["True", "False"],
                                   "Ignore index": -100,
                                   "Reduce": ["True", "False"],
                                   "Reduction": ["mean", "sum", "none"]}
        elif caller == "Loss Function:MSE Loss" and caller not in self.params:
            self.params[caller] = {"Size average": ["True", "False"],
                                   "Reduce": ["True", "False"],
                                   "Reduction": ["mean", "sum", "none"]}
        elif caller == "Metrics:Inaccuracy" and caller not in self.params:
            self.params[caller] = {}
        elif caller == "Metrics:MSE Loss" and caller not in self.params:
            self.params[caller] = {"Size average": ["True", "False"],
                                   "Reduce": ["True", "False"],
                                   "Reduction": ["mean", "sum", "none"]}

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
        for k, v in self.params[name].items():
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

        param = self.params[name][key]

        # Update list putting the selected value at index 0
        if type(param) == list:
            param.remove(value)
            self.params[name][key] = [value] + param
        else:
            self.params[name][key] = value
