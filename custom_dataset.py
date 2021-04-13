import os

import torchvision
from PIL import Image
from torchvision import transforms

from config import *
from models.custom import Custom
from utils import *


def test_custom_data(verbose=False, visualize=False, folder='custom_data'):
    """Function capable of running standalone to evaluate model accuracy"""
    if visualize:
        verbose = True

    # Standardize and load test data
    transform = transforms.Compose(
        [transforms.Resize(200),
         transforms.CenterCrop(200),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.ImageFolder(f'{folder}', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)

    # Initialize and load saved model
    net = Custom()
    net.load_state_dict(torch.load(PATH))

    # If verbose mode is set, show model output for each image
    if verbose:
        predictions = []
        files = []
        for directory in os.listdir(folder):
            for file in os.listdir(f'{folder}/{directory}'):
                img = Image.open(f'{folder}/{directory}/{file}').convert('RGB')
                files.append((file, img, directory))
        for file in files:
            img = transform(file[1])
            img = torch.unsqueeze(img, 0)
            with torch.no_grad():
                _, predicted = torch.max(net(img).data, 1)
                if predicted.data[0].item() == 1:
                    predictions.append('pedestrian')
                else:
                    predictions.append('not_pedestrian')
        for i in range(len(files)):
            print(
                f'{files[i][2]}/{files[i][0]}: pred={predictions[i]}, actu={files[i][2]}, {"correct" if predictions[i] == files[i][2] else "incorrect"}')
            if visualize:
                plt.imshow(files[i][1])
                plt.title(predictions[i])
                plt.show()
            i += 1

    # Evaluate model accuracy on the test data
    evaluate(net, testloader, 'Test Acc |')


if __name__ == '__main__':
    test_custom_data()
