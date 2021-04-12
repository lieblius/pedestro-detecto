import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from utils import *


class CIFAR100People(CIFAR100):
    """Unused in current implementation. Torchvision CIFAR100 class with labels consolidated down to binary person or
    not_person """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100People, self).__init__(root, train, transform, target_transform, download)

        coarse_labels = np.array(
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.targets = coarse_labels[self.targets]

        self.classes = [
            ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
             'orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'bottle', 'bowl', 'can', 'cup', 'plate', 'apple',
             'mushroom', 'orange', 'pear', 'sweet_pepper', 'clock', 'keyboard', 'lamp', 'telephone', 'television',
             'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
             'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud',
             'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox',
             'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'crocodile',
             'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple_tree',
             'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'bicycle', 'bus', 'motorcycle', 'pickup_truck',
             'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
            ['baby', 'boy', 'girl', 'man', 'woman']]


def get_cifar_data(download_enabled=False):
    """Unused in current implementation. Loads CIFAR100People training and test data and classes"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR100People(root='./cifar_data', train=True,
                              download=download_enabled, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = CIFAR100People(root='./cifar_data', train=False,
                             download=download_enabled, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)

    classes = ['not_person', 'person']

    return trainloader, testloader, classes


def get_custom_data(path):
    """Loads custom data from path and standardizes to 200x200 normalized tensors"""
    transform = transforms.Compose(
        [transforms.Resize(200),
         transforms.CenterCrop(200),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(f'{path}/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.ImageFolder(f'{path}/validation', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=0)

    return trainloader, testloader
