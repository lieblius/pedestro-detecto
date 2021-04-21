import os

from config import PATH, OVERWRITE
from custom_dataset import test_custom_data
from dataset import get_custom_data
from models.custom import Custom
from train_net import train
from utils import *


def main():
    # Load training and validation datas
    trainloader, valloader = get_custom_data('data')

    # Initialize network and visualize
    net = Custom()
    viewmodel(net)

    # If model is not already trained or to be overwritten then train, otherwise load and evaluate best epoch
    if not os.path.isfile(PATH) or OVERWRITE:
        train(trainloader, net, valloader, num_epochs=15)
    else:
        net.load_state_dict(torch.load(PATH))
        evaluate(net, trainloader, 'Train Acc |')
        evaluate(net, valloader, 'Val Acc |')

    # Test model on test set containing outside data not from the same dataset
    # Custom data layout:
    #  custom_data/
    #     |
    #     ----->not_pedestrian/
    #     |           |
    #     |           -----> *.jpg/png
    #     ------>pedestrian/
    #                |
    #                ------> *.jpg/png
    test_custom_data(verbose=True, visualize=True, folder='custom_data')


if __name__ == '__main__':
    main()
