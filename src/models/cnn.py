"""
Custom CNN for CIFAR-10/100 — used as the primary model for all experiments.

Architecture: 2 conv + 5 FC layers, ~200K parameters. The depth (many small FC
layers) is intentional — it gives FedPart enough layers to demonstrate cycling
behaviour without being so big that it dominates the experimental compute budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Custom CNN.

    Layer index → name (used by partial training):
        0 conv1.weight  / 1 conv1.bias
        2 conv2.weight  / 3 conv2.bias
        4 fc1.weight    / 5 fc1.bias
        6 fc2.weight    / 7 fc2.bias
        ...
        14 fc7.weight  / 15 fc7.bias
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 180)
        self.fc2 = nn.Linear(180, 160)
        self.fc3 = nn.Linear(160, 140)
        self.fc4 = nn.Linear(140, 120)
        self.fc5 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc7(x)
        return x


def num_layer_pairs(model: nn.Module) -> int:
    """Number of (weight, bias) pairs in the model — used by layer-wise strategies."""
    return len(list(model.parameters())) // 2
