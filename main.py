import pickle
import torch
import torchvision
import numpy as np
import os

from config import *
from dataset import get_data
from models.net import Net
from train_net import train
from test_net import test
from utils import *


def main():
    trainloader, testloader, classes = get_data(False)

    net = Net()
    if not os.path.isfile(PATH):
        train(trainloader, net)
    else:
        net.load_state_dict(torch.load(PATH))

    test(testloader, net, classes)


if __name__ == '__main__':
    main()
