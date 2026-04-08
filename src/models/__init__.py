"""Model factory."""

import torch.nn as nn

from .cnn import CNN, num_layer_pairs


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build a model by name."""
    if name == "cnn":
        return CNN(num_classes=num_classes)
    if name == "resnet18":
        from .resnet18_gn import resnet18_gn
        return resnet18_gn(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["build_model", "CNN", "num_layer_pairs"]
