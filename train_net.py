from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import PATH
from utils import evaluate


def train(trainloader, net, valloader, num_epochs=10):
    # Set loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Begin training
    train_accuracy, validation_accuracy, train_loss, models = [], [], [], []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, bar_format=f'Epoch {epoch} |' + '{l_bar}{bar:10}{r_bar}{bar:-10b}'),
                                 0):
            inputs, labels = data
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Running Loss: {running_loss}')
        train_accuracy.append(evaluate(net, trainloader, 'Train Acc |'))
        validation_accuracy.append(evaluate(net, valloader, 'Val Acc |'))
        train_loss.append(running_loss)
        models.append(deepcopy(net))
        print('----------------------------------------------------------')

    print('Finished Training')

    # Select model with highest validation accuracy
    net = deepcopy(models[np.argmax(validation_accuracy)])

    # Save model
    torch.save(net.state_dict(), PATH)

    # Save visualizations
    plt.figure()
    x = np.arange(num_epochs)
    plt.subplot(2, 1, 1)
    plt.plot(x, train_accuracy)
    plt.plot(x, validation_accuracy)
    plt.legend(['Training', 'Validation'])
    plt.xticks(x)
    plt.ylabel('Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x, train_loss)
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.savefig('training_visualization.png')
