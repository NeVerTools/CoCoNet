import numpy as np

import never2.core.controller.pynevertemp.datasets as dt
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import never2.core.controller.pynevertemp.networks as networks
import never2.core.controller.pynevertemp.nodes as nodes
import never2.core.controller.pynevertemp.strategies.training as training
import torch.optim as opt
import torch.optim.lr_scheduler as schedulers
import torch.nn as nn
import torch.nn.functional as funct
import torch

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Lambda(lambda x: torch.flatten(x))])

fmnist = dt.TorchFMNIST("data/", True, transform)

network = networks.SequentialNetwork("TEST1", "X")
fc1 = nodes.FullyConnectedNode("FC1", (784,), 128)
network.add_node(fc1)
rl2 = nodes.ReLUNode("RL2", fc1.out_dim)
network.add_node(rl2)
fc3 = nodes.FullyConnectedNode("FC3", rl2.out_dim, 64)
network.add_node(fc3)
rl4 = nodes.ReLUNode("RL4", fc3.out_dim)
network.add_node(rl4)
fc5 = nodes.FullyConnectedNode("FC5", rl4.out_dim, 10)
network.add_node(fc5)

tr = training.PytorchTraining(opt.Adam, dict(), schedulers.ReduceLROnPlateau, dict(), funct.cross_entropy, dict(),
                              training.PytorchMetrics.inaccuracy, dict(), 3, 0.2, 512, 64)

tr.train(network, fmnist)
