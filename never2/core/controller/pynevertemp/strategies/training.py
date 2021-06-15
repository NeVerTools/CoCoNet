import abc
import never2.core.controller.pynevertemp.networks as networks
import never2.core.controller.pynevertemp.datasets as datasets
import never2.core.controller.pynevertemp.strategies.conversion as cv
import os
import shutil
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as funct
import never2.core.controller.pynevertemp.utilities as utilities


class TrainingStrategy(abc.ABC):
    """
    An abstract class used to represent a Training Strategy.

    Methods
    ----------
    train(NeuralNetwork, Dataset)
        Train the neural network of interest using a training strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def train(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Train the neural network of interest using a pruning strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        dataset : Dataset
            The dataset to use to train the neural network.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the training of the original network using the training strategy and the
            dataset.

        """
        pass


class AdamTraining(TrainingStrategy):
    """
    A concrete class used to represent the Adam training strategy.
    This kind of training is based on an Adam optimizer.
    We refer to https://arxiv.org/abs/1412.6980 for theoretical details on the optimization algorithm.

    Attributes
    ----------
    n_epochs : int
        Number of epochs for the training procedure.
    train_batch_size : int
        Dimension for the train batch size for the training procedure
    test_batch_size : int
        Dimension for the test batch size for the training procedure
    learning_rate : float
        Learning rate parameter for the fine tuning procedure.
    betas : Tuple (float, float), optional
        Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
    eps : float, optional
        Term added to the denominator to improve numerical stability (default: 1e-8).
    weight_decay : float, optional
        Coefficient of the L2 norm regularizer of the Adam optimizer (default: 0).
    cuda : bool, optional
        Whether to use the cuda library for the procedure (default: False).
    train_patience : int, optional
        The number of epochs in which the loss may not decrease before the
        training procedure is interrupted (default: 10).
    scheduler_patience : int, optional
        The number of epochs in which the loss may not decrease before the
        scheduler decrease the learning rate (default: 3).
    batchnorm_decay : float, optional
        It is the coefficient of the L1 norm regularizer applied only to the weights of the batch
        normalization layers. It is a preparatory parameter for pruning strategies which leverages
        the coefficients of the batch normalization layers. It should be selected considering the
        sparsity rate of the related pruning procedure (default: 0).
    l1_decay: float, optional
        Coefficient of the L1 norm regularizer. It should not be used with the weight_decay regularizer.
        It is also a preparatory parameter for pruning strategies which leverages the near-to-zero value of
        the weights (default: 0).
    fine_tuning : bool, optional
        Whether the training procedure should use the fine tuning routine (i.e., the weight with value = 0 are
        not updated by the optimizer (default: False).

    Methods
    ----------
    train(NeuralNetwork, Dataset)
        Train the neural network of interest using the training strategy Adam Training and the dataset passed as an
        argument.

    """

    def __init__(self, n_epochs: int, train_batch_size: int, test_batch_size: int, learning_rate: float,
                 betas: (float, float) = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0, cuda: bool = False,
                 train_patience: int = 10, scheduler_patience: int = 3, batchnorm_decay: float = 0,
                 l1_decay: float = 0, fine_tuning: bool = False):

        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.cuda = cuda
        self.train_patience = train_patience
        self.scheduler_patience = scheduler_patience
        self.batchnorm_decay = batchnorm_decay
        self.l1_decay = l1_decay
        self.fine_tuning = fine_tuning

    def train(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Train the neural network of interest using the training strategy SGD Training.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        dataset : Dataset
            The dataset to use for the training of the network.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the training strategy to the original network.

        """

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__training(py_net, dataset)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        return network

    def __training(self, net: cv.PyTorchNetwork, dataset: datasets.Dataset) -> cv.PyTorchNetwork:

        """
        Training procedure for the PyTorchNetwork.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to train.
        dataset : Dataset
            The dataset to use for the training of the PyTorchNetwork

        Returns
        ----------
        PyTorchNetwork
            The trained PyTorchNetwork.

        """

        # If the training should be done with the GPU we set the model to cuda.
        if self.cuda:
            net.pytorch_network.cuda()
        else:
            net.pytorch_network.cpu()

        net.pytorch_network.train()
        net.pytorch_network.float()

        # We define the optimizer and the scheduler with the correct parameters.
        optimizer = torch.optim.Adam(params=net.pytorch_network.parameters(), lr=self.learning_rate, betas=self.betas,
                                     eps=self.eps, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_patience)

        start_epoch = 0
        training_set = dataset.get_training_set()

        # If a checkpoint exist, we load the checkpoint of interest
        # checkpoints_path = 'checkpoints/' + net.identifier + '.pth.tar'
        # best_model_path = 'checkpoints/' + net.identifier + '_best.pth.tar'

        checkpoints_path = net.identifier + '.pth.tar'
        best_model_path = net.identifier + '_best.pth.tar'

        if os.path.isfile(checkpoints_path):

            print("=> loading checkpoint '{}'".format(checkpoints_path))
            checkpoint = torch.load(checkpoints_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.pytorch_network.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_loss = checkpoint['best_val_loss']
            epochs_without_decrease = checkpoint['no_dec_epochs']
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(checkpoints_path, checkpoint['epoch'], best_prec1))

        else:
            print("=> no checkpoint found at '{}'".format(checkpoints_path))
            best_val_loss = 999
            epochs_without_decrease = 0

        history_score = np.zeros((self.n_epochs - start_epoch + 1, 3))

        # TRAINING

        best_prec1 = 0
        for epoch in range(start_epoch, self.n_epochs):

            if epochs_without_decrease > self.train_patience:
                break

            # EPOCH TRAINING

            net.pytorch_network.train()
            avg_loss = 0
            train_acc = 0
            batch_idx = 0
            data_idx = 0

            while data_idx < len(training_set[0]):

                if data_idx + self.train_batch_size >= len(training_set[0]):
                    last_data_idx = len(training_set[0])
                else:
                    last_data_idx = data_idx + self.train_batch_size

                data = torch.from_numpy(training_set[0][data_idx:last_data_idx, :])
                target = torch.from_numpy(training_set[1][data_idx:last_data_idx])

                if self.cuda:
                    data, target = data.cuda(), target.cuda()

                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                optimizer.zero_grad()
                output = net.pytorch_network(data)
                loss = funct.cross_entropy(output, target)
                avg_loss += loss.data.item()
                pred = output.data.max(1, keepdim=True)[1]
                train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
                loss.backward()

                # Pruning oriented training: it regularizes the batch norm coef values in order to identify unimportant
                # channels.
                for m in net.pytorch_network.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.weight.grad.data.add_(self.batchnorm_decay * torch.sign(m.weight.data))

                # Pruning oriented training: it regularizes the weights in order to identify the unimportant ones.
                for m in net.pytorch_network.modules():

                    if isinstance(m, nn.Linear):
                        m.weight.grad.data.add_(self.l1_decay * torch.sign(m.weight.data))

                # If the fine_tuning flag is set then we assume that the training is used as fine tuning for a
                # weight pruning procedure, therefore the weight with value = 0 are not updated.
                if self.fine_tuning:

                    for m in net.pytorch_network.modules():

                        if isinstance(m, nn.Linear):
                            weight_copy = m.weight.data.abs().clone()
                            if self.cuda:
                                mask = weight_copy.gt(0).float().cuda()
                            else:
                                mask = weight_copy.gt(0).float()
                            m.weight.grad.data.mul_(mask)

                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(training_set[0]),
                               100. * batch_idx / math.floor(len(training_set[0]) / self.train_batch_size),
                        loss.data.item()))

                data_idx += self.train_batch_size
                batch_idx += 1

            history_score[epoch - start_epoch][0] = avg_loss / float(math.floor(len(training_set[0]) /
                                                                                self.train_batch_size))
            history_score[epoch - start_epoch][1] = train_acc / float(math.floor(len(training_set[0]) /
                                                                                 self.train_batch_size))

            # EPOCH TEST

            prec1, test_loss = utilities.testing(net, dataset, self.test_batch_size, self.cuda)

            if test_loss < best_val_loss:
                epochs_without_decrease = 0
                best_val_loss = test_loss
            else:
                epochs_without_decrease += 1

            if scheduler is not None:
                scheduler.step(test_loss)

            # CHECKPOINT

            history_score[epoch - start_epoch][2] = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            state = {
                'epoch': epoch + 1,
                'state_dict': net.pytorch_network.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'no_dec_epochs': epochs_without_decrease,
            }
            torch.save(state, checkpoints_path)
            if is_best:
                shutil.copyfile(checkpoints_path, best_model_path)

        print("Best accuracy: " + str(best_prec1))
        history_score[-1][0] = best_prec1

        if os.path.isfile(checkpoints_path):
            os.remove(checkpoints_path)

        if os.path.isfile(best_model_path):
            os.remove(best_model_path)

        return net


class AdamTrainingRegression(TrainingStrategy):
    """
    A concrete class used to represent the Adam training strategy for regression.
    This kind of training is based on an Adam optimizer.
    We refer to https://arxiv.org/abs/1412.6980 for theoretical details on the optimization algorithm.

    Attributes
    ----------
    n_epochs : int
        Number of epochs for the training procedure.
    train_batch_size : int
        Dimension for the train batch size for the training procedure
    test_batch_size : int
        Dimension for the test batch size for the training procedure
    learning_rate : float
        Learning rate parameter for the fine tuning procedure.
    betas : Tuple (float, float), optional
        Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
    eps : float, optional
        Term added to the denominator to improve numerical stability (default: 1e-8).
    weight_decay : float, optional
        Coefficient of the L2 norm regularizer of the Adam optimizer (default: 0).
    cuda : bool, optional
        Whether to use the cuda library for the procedure (default: False).
    train_patience : int, optional
        The number of epochs in which the loss may not decrease before the
        training procedure is interrupted (default: 10).
    scheduler_patience : int, optional
        The number of epochs in which the loss may not decrease before the
        scheduler decrease the learning rate (default: 3).

    Methods
    ----------
    train(NeuralNetwork, Dataset)
        Train the neural network of interest using the training strategy Adam Training and the dataset passed as an
        argument.

    """

    def __init__(self, n_epochs: int, train_batch_size: int, test_batch_size: int, learning_rate: float,
                 betas: (float, float) = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0, cuda: bool = False,
                 train_patience: int = 10, scheduler_patience: int = 3):

        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.cuda = cuda
        self.train_patience = train_patience
        self.scheduler_patience = scheduler_patience

    def train(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Train the neural network of interest using the training strategy SGD Training.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        dataset : Dataset
            The dataset to use for the training of the network.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the training strategy to the original network.

        """

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__training(py_net, dataset)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        return network

    def __training(self, net: cv.PyTorchNetwork, dataset: datasets.Dataset) -> cv.PyTorchNetwork:

        """
        Training procedure for the PyTorchNetwork.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to train.
        dataset : Dataset
            The dataset to use for the training of the PyTorchNetwork

        Returns
        ----------
        PyTorchNetwork
            The trained PyTorchNetwork.

        """

        # If the training should be done with the GPU we set the model to cuda.
        if self.cuda:
            net.pytorch_network.cuda()
        else:
            net.pytorch_network.cpu()

        net.pytorch_network.train()
        net.pytorch_network.double()

        # We define the optimizer and the scheduler with the correct parameters.
        optimizer = torch.optim.Adam(params=net.pytorch_network.parameters(), lr=self.learning_rate, betas=self.betas,
                                     eps=self.eps, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_patience)

        start_epoch = 0
        training_set = dataset.get_training_set()

        # If a checkpoint exist, we load the checkpoint of interest
        # checkpoints_path = 'checkpoints/' + net.identifier + '.pth.tar'
        # best_model_path = 'checkpoints/' + net.identifier + '_best.pth.tar'

        checkpoints_path = net.identifier + '.pth.tar'
        best_model_path = net.identifier + '_best.pth.tar'

        if os.path.isfile(checkpoints_path):

            print("=> loading checkpoint '{}'".format(checkpoints_path))
            checkpoint = torch.load(checkpoints_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.pytorch_network.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_loss = checkpoint['best_val_loss']
            epochs_without_decrease = checkpoint['no_dec_epochs']
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(checkpoints_path, checkpoint['epoch'], best_prec1))

        else:
            print("=> no checkpoint found at '{}'".format(checkpoints_path))
            best_val_loss = 999
            epochs_without_decrease = 0

        history_score = np.zeros((self.n_epochs - start_epoch + 1, 3))

        # TRAINING

        best_prec1 = 999
        for epoch in range(start_epoch, self.n_epochs):

            if epochs_without_decrease > self.train_patience:
                break

            # EPOCH TRAINING

            net.pytorch_network.train()
            avg_loss = 0
            train_acc = 0
            batch_idx = 0
            data_idx = 0

            while data_idx < len(training_set[0]):

                if data_idx + self.train_batch_size >= len(training_set[0]):
                    last_data_idx = len(training_set[0])
                else:
                    last_data_idx = data_idx + self.train_batch_size

                data = torch.from_numpy(training_set[0][data_idx:last_data_idx, :])
                target = torch.from_numpy(training_set[1][data_idx:last_data_idx])

                if self.cuda:
                    data, target = data.cuda(), target.cuda()

                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                data = data.double()
                target = target.double()
                optimizer.zero_grad()
                output = net.pytorch_network(data)
                loss = funct.mse_loss(output, target)
                avg_loss += loss.data.item()
                loss.backward()

                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(training_set[0]),
                               100. * batch_idx / math.floor(len(training_set[0]) / self.train_batch_size),
                        loss.data.item()))

                data_idx += self.train_batch_size
                batch_idx += 1

            history_score[epoch - start_epoch][0] = avg_loss / float(math.floor(len(training_set[0]) /
                                                                                self.train_batch_size))
            history_score[epoch - start_epoch][1] = train_acc / float(math.floor(len(training_set[0]) /
                                                                                 self.train_batch_size))

            # EPOCH TEST

            test_set = dataset.get_test_set()

            net.pytorch_network.eval()
            net.pytorch_network.double()
            test_loss = 0
            with torch.no_grad():

                batch_idx = 0
                data_idx = 0

                while data_idx < len(test_set[0]):

                    if data_idx + self.test_batch_size >= len(test_set[0]):
                        last_data_idx = len(test_set[0])
                    else:
                        last_data_idx = data_idx + self.test_batch_size

                    data = torch.from_numpy(test_set[0][data_idx:last_data_idx, :])
                    target = torch.from_numpy(test_set[1][data_idx:last_data_idx])

                    if self.cuda:
                        data, target = data.cuda(), target.cuda()

                    data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                    data = data.double()
                    target = target.double()
                    output = net.pytorch_network(data)
                    test_loss += funct.mse_loss(output, target).data.item()  # sum up batch loss
                    batch_idx += 1
                    data_idx += self.test_batch_size

            test_loss /= float(math.floor(len(test_set[0]) / self.test_batch_size))
            print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

            if test_loss < best_val_loss:
                epochs_without_decrease = 0
                best_val_loss = test_loss
            else:
                epochs_without_decrease += 1

            if scheduler is not None:
                scheduler.step(test_loss)

            # CHECKPOINT

            history_score[epoch - start_epoch][2] = test_loss
            is_best = test_loss < best_prec1
            best_prec1 = min(test_loss, best_prec1)

            state = {
                'epoch': epoch + 1,
                'state_dict': net.pytorch_network.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'no_dec_epochs': epochs_without_decrease,
            }

            torch.save(state, checkpoints_path)
            if is_best:
                shutil.copyfile(checkpoints_path, best_model_path)

        print("Best Loss: " + str(best_prec1))
        history_score[-1][0] = best_prec1

        if os.path.isfile(checkpoints_path):
            os.remove(checkpoints_path)

        if os.path.isfile(best_model_path):
            os.remove(best_model_path)

        return net
