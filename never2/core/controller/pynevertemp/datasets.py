import abc
import numpy as np
from never2.core.controller.pynevertemp.tensor import Tensor
import torchvision as tv
import torch


class Dataset(abc.ABC):
    """
    An abstract class used to represent a Dataset.

    Methods
    ----------
    get_training_set()
        Return the part of the dataset used as the Training set as a tuple of Tensors (Data and Targets).
        It must be implemented in the concrete classes.
    get_test_set()
        Return the part of the dataset used as the Test set as a tuple of Tensors (Data and Targets).
        It must be implemented in the concrete classes.
    add_training_sample((Tensor, Tensor))
        Add a new sample to the training set.
        It must be implemented in the concrete classes.
    add_test_sample((Tensor, Tensor))
        Add a new sample to the test set.
        It must be implemented in the concrete classes.

    """

    @abc.abstractmethod
    def get_training_set(self) -> (Tensor, Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_set(self) -> (Tensor, Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def add_training_sample(self, sample: (Tensor, Tensor)):
        raise NotImplementedError

    @abc.abstractmethod
    def add_test_sample(self, sample: (Tensor, Tensor)):
        raise NotImplementedError


class MNISTDataset(Dataset):
    """
    A concrete class used to represent the MNIST Dataset.

    Attributes
    ----------
    training_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the training set.
    test_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the test set.

    Methods
    ----------
    get_training_set()
        Return the part of the dataset used as the Training set as a tuple of Tensors (Data and Targets).
    get_test_set()
        Return the part of the dataset used as the Test set as a tuple of Tensors (Data and Targets).
    add_training_sample((Tensor, Tensor))
        Add a new sample to the training set.
        It must be implemented in the concrete classes.
    add_test_sample((Tensor, Tensor))
        Add a new sample to the test set.
        It must be implemented in the concrete classes.
    """

    def __init__(self):

        datapath = 'data/'

        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ])

        train_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST(datapath, train=True, download=True, transform=transform), batch_size=1,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST(datapath, train=False, transform=transform), batch_size=1, shuffle=True)

        data_list = []
        target_list = []
        for data, target in train_loader:
            data_np = data[0][0].view(-1).numpy()
            target_np = target.numpy()
            data_list.append(data_np)
            target_list.append(target_np)

        self.training_set = (np.array(data_list), np.array(target_list).reshape(-1))

        data_list = []
        target_list = []
        for data, target in test_loader:
            data_np = data[0][0].view(-1).numpy()
            target_np = target.numpy()
            data_list.append(data_np)
            target_list.append(target_np)

        self.test_set = (np.array(data_list), np.array(target_list).reshape(-1))

    def get_training_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Training Set of the MNIST dataset: such set contains 60000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The training set as a tuple of Tensor.
        """
        return self.training_set

    def get_test_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Test Set of the MNIST dataset: such set contains 10000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The test set as a tuple of Tensor.
        """
        return self.test_set

    def add_training_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.training_set = (np.append(self.training_set[0], sample[0], axis=0),
                             np.append(self.training_set[1], sample[1]))

    def add_test_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.test_set = (np.append(self.test_set[0], sample[0], axis=0),
                         np.append(self.test_set[1], sample[1]))


class FMNISTDataset(Dataset):
    """
    A concrete class used to represent the FMNIST Dataset.

    Attributes
    ----------
    training_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the training set.
    test_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the test set.

    Methods
    ----------
    get_training_set()
        Return the part of the dataset used as the Training set as a tuple of Tensors (Data and Targets).
    get_test_set()
        Return the part of the dataset used as the Test set as a tuple of Tensors (Data and Targets).
    add_training_sample((Tensor, Tensor))
        Add a new sample to the training set.
        It must be implemented in the concrete classes.
    add_test_sample((Tensor, Tensor))
        Add a new sample to the test set.
        It must be implemented in the concrete classes.
    """

    def __init__(self):

        datapath = 'data/'

        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ])

        train_loader = torch.utils.data.DataLoader(
            tv.datasets.FashionMNIST(datapath, train=True, download=True, transform=transform), batch_size=1,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            tv.datasets.FashionMNIST(datapath, train=False, transform=transform), batch_size=1, shuffle=True)

        data_list = []
        target_list = []
        for data, target in train_loader:
            data_np = data[0][0].view(-1).numpy()
            target_np = target.numpy()
            data_list.append(data_np)
            target_list.append(target_np)

        self.training_set = (np.array(data_list), np.array(target_list).reshape(-1))

        data_list = []
        target_list = []
        for data, target in test_loader:
            data_np = data[0][0].view(-1).numpy()
            target_np = target.numpy()
            data_list.append(data_np)
            target_list.append(target_np)

        self.test_set = (np.array(data_list), np.array(target_list).reshape(-1))

    def get_training_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Training Set of the MNIST dataset: such set contains 60000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The training set as a tuple of Tensor.
        """
        return self.training_set

    def get_test_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Test Set of the MNIST dataset: such set contains 10000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The test set as a tuple of Tensor.
        """
        return self.test_set

    def add_training_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.training_set = (np.append(self.training_set[0], sample[0], axis=0),
                             np.append(self.training_set[1], sample[1]))

    def add_test_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.test_set = (np.append(self.test_set[0], sample[0], axis=0),
                         np.append(self.test_set[1], sample[1]))


class DynamicsJamesPos(Dataset):
    """
    A concrete class used to represent the Dynamic James Dataset presented in the paper
    "Challenging SMT solvers to verify neural networks" by Pulina and Tacchella (2012).

    Attributes
    ----------
    training_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the training set.
    test_set : (Tensor, Tensor)
        Tuple of Tensors containing the data and the target of the part of the dataset forming the test set.

    Methods
    ----------
    get_training_set()
        Return the part of the dataset used as the Training set as a tuple of Tensors (Data and Targets).
    get_test_set()
        Return the part of the dataset used as the Test set as a tuple of Tensors (Data and Targets).
    add_training_sample((Tensor, Tensor))
        Add a new sample to the training set.
        It must be implemented in the concrete classes.
    add_test_sample((Tensor, Tensor))
        Add a new sample to the test set.
        It must be implemented in the concrete classes.
    """

    def __init__(self, training_set_path: str, test_set_path: str):

        train_set = np.loadtxt(training_set_path, delimiter=",")
        t_set = np.loadtxt(test_set_path, delimiter=",")
        training_set = (train_set[:, 0:8], train_set[:, 8:])
        test_set = (t_set[:, 0:8], t_set[:, 8:])
        self.training_set = training_set
        self.test_set = test_set

    def get_training_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Training Set of the MNIST dataset: such set contains 60000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The training set as a tuple of Tensor.
        """
        return self.training_set

    def get_test_set(self) -> (Tensor, Tensor):
        """
        Procedure which returns the Test Set of the MNIST dataset: such set contains 10000 grayscale images.

        Returns
        -------
        (Tensor, Tensor)
            The test set as a tuple of Tensor.
        """
        return self.test_set

    def add_training_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.training_set = (np.append(self.training_set[0], sample[0], axis=0),
                             np.append(self.training_set[1], sample[1]))

    def add_test_sample(self, sample: (Tensor, Tensor)):
        """
        Procedure which adds a new sample to the Training Set of the MNIST dataset.

        Parameters
        ----------
        sample : (Tensor, Tensor)
            The sample we wish to add to the set. The first element is the data and the second is the target.

        """
        self.test_set = (np.append(self.test_set[0], sample[0], axis=0),
                         np.append(self.test_set[1], sample[1]))
