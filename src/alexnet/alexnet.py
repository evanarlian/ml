import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),  # overlapping pooling from paper
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),  # overlapping pooling from paper
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),  # overlapping pooling from paper
            nn.Flatten(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout1d(p=0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout1d(p=0.5),
        )
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.middle(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def debug_forward(self, x):
        # fmt: off
        x = self.conv1(x); print("after conv1", x.size())
        x = self.conv2(x); print("after conv2", x.size())
        x = self.conv3(x); print("after conv3", x.size())
        x = self.conv4(x); print("after conv4", x.size())
        x = self.conv5(x); print("after conv5", x.size())
        x = self.middle(x); print("after middle", x.size())
        x = self.fc1(x); print("after fc1", x.size())
        x = self.fc2(x); print("after fc2", x.size())
        x = self.fc3(x); print("after fc3", x.size())
        # fmt: off
        return x
