from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QGridLayout, QLineEdit

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

        # Main body
        body_layout = QHBoxLayout()
        params_layout = QGridLayout()

        self.params["Optimizer"] = QComboBox()
        self.params["Optimizer"].addItem("Adam (pyTorch)")
        self.params["Optimizer"].activated.connect(lambda: self.details_layout.update_self("Optimizer"))

        self.params["Scheduler"] = QComboBox()
        self.params["Scheduler"].addItem("ReduceLROnPlateau (pyTorch)")
        self.params["Scheduler"].activated.connect(lambda: self.details_layout.update_self("Scheduler"))

        self.params["Loss Function"] = QComboBox()
        self.params["Loss Function"].addItems(["Cross Entropy (pyTorch)",
                                               "MSE Loss (pyTorch)"])
        self.params["Loss Function"].activated.connect(lambda: self.details_layout.update_self("Loss Function"))

        self.params["Metrics"] = QComboBox()
        self.params["Metrics"].addItems(["Inaccuracy",
                                         "MSE Loss"])
        self.params["Metrics"].activated.connect(lambda: self.details_layout.update_self("Metrics"))

        self.params["Transform"] = QComboBox()
        self.params["Transform"].addItems(["pil_to_tensor (pyTorch)",
                                           "Normalize (pyTorch)"])
        self.params["Transform"].activated.connect(lambda: self.details_layout.update_self("Transform"))

        params_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        params_layout.addWidget(self.params["Optimizer"], 0, 1)
        params_layout.addWidget(QLabel("Scheduler:"), 1, 0)
        params_layout.addWidget(self.params["Scheduler"], 1, 1)
        params_layout.addWidget(QLabel("Loss Function:"), 2, 0)
        params_layout.addWidget(self.params["Loss Function"], 2, 1)
        params_layout.addWidget(QLabel("Metrics:"), 3, 0)
        params_layout.addWidget(self.params["Metrics"], 3, 1)
        params_layout.addWidget(QLabel("Transform:"), 4, 0)
        params_layout.addWidget(self.params["Transform"], 4, 1)
        body_layout.addLayout(params_layout)

        self.details_layout = GUIParamLayout()
        self.details_layout.setAlignment(Qt.AlignTop)
        body_layout.addLayout(self.details_layout)
        self.layout.addLayout(body_layout)

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
        self.addLayout(self.grid_layout)

    def update_self(self, caller: str):
        self.clear_grid()

        if caller == "Optimizer":
            self.show_opt_layout()

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

    def show_opt_layout(self):
        lr_input = QLineEdit("1e-3")
        rate_pair = (QLabel("Learning rate"), lr_input)
        self.grid_dict["lf"] = rate_pair
        betas_input = QLineEdit("(0.9, 0.999)")
        betas_pair = (QLabel("Betas"), betas_input)
        self.grid_dict["betas"] = betas_pair
        eps_input = QLineEdit("1e-8")
        eps_pair = (QLabel("Epsilon"), eps_input)
        self.grid_dict["eps"] = eps_pair
        wdec_input = QLineEdit("0")
        wdec_pair = (QLabel("Weight decay"), wdec_input)
        self.grid_dict["weight_decay"] = wdec_pair
        amsgr_input = QComboBox()
        amsgr_input.addItems(["False", "True"])
        amsgr_pair = (QLabel("AMSGrad"), amsgr_input)
        self.grid_dict["amsgrad"] = amsgr_pair

        count = 1
        for element in self.grid_dict.values():
            self.grid_layout.addWidget(element[0], count, 0)
            self.grid_layout.addWidget(element[1], count, 1)
            count += 1
