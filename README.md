# pedestro-detecto
CNN for binary image classification of pedestrians.

## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 100, 100]           2,432
         MaxPool2d-2           [-1, 32, 25, 25]               0
            Conv2d-3          [-1, 128, 13, 13]         102,528
         MaxPool2d-4            [-1, 128, 3, 3]               0
            Conv2d-5             [-1, 32, 2, 2]         102,432
            Linear-6                   [-1, 16]           2,064
            Linear-7                    [-1, 2]              34
================================================================
Total params: 209,490
```

## Current Results:
```
Training Accuracy: 100%
Validation Accuracy: 93.5%
Test Accuracy: 73% on "testdata/Road with Person" and "testdata/Road without Person", 100% on "data"
```

## Resources:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://pytorch.org/vision/stable/datasets.html#cifar
https://www.cs.toronto.edu/~kriz/cifar.html
https://github.com/bearpaw/pytorch-classification
https://gist.github.com/beeva-albertorincon/1ef96e071ac5adcb421663f3bbe7b1a6
https://github.com/ryanchankh/cifar100coarse/
https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc
https://www.kaggle.com/tejasvdante/pedestrian-no-pedestrian

Tested on Python 3.7.10
