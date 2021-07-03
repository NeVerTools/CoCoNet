import logging

import torch
import torch.nn.functional as funct
import torch.optim as opt
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms

import coconet.core.controller.pynevertemp.datasets as dt
import coconet.core.controller.pynevertemp.networks as networks
import coconet.core.controller.pynevertemp.nodes as nodes
import coconet.core.controller.pynevertemp.strategies.training as training

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

logger = logging.getLogger("pynever")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

tr = training.PytorchTraining(opt.Adam, dict(), funct.cross_entropy, 3, 0.2, 512, 64,
                              schedulers.ReduceLROnPlateau, dict(),
                              training.PytorchMetrics.inaccuracy)

tr.train(network, fmnist)
