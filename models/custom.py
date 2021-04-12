from math import sqrt

import torch.nn as nn
import torch.nn.functional as F


class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5), (2, 2), (2, 2))
        self.pool = nn.MaxPool2d((4, 4), (4, 4))
        self.conv2 = nn.Conv2d(32, 128, (5, 5), (2, 2), (2, 2))
        self.conv3 = nn.Conv2d(128, 32, (5, 5), (2, 2), (2, 2))
        self.fc_1 = nn.Linear(32*4, 16)
        self.fc_2 = nn.Linear(16, 2)
        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_2.bias, 0.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        N, C, H, W = x.shape
        x = self.fc_1(x.view(N, C * H * W))
        x = self.fc_2(x)
        return x
