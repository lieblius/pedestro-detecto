import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary
from tqdm import tqdm


def viewmodel(net):
    summary(net, (3, 200, 200))


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


def imshow(img):
    plt.imshow(np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0)))
    plt.show()


def evaluate(model, loader, title=''):
    device = torch.device('cpu')
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, label in tqdm(loader, bar_format=f'{title}' + '{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
        acc = correct / len(loader.dataset)
        print("Evaluation accuracy: {}".format(acc))
        return acc
