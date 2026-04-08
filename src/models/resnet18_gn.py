"""
ResNet-18 with GroupNorm replacing BatchNorm for federated learning compatibility.

BatchNorm is incompatible with FL because it depends on batch statistics
that differ across clients. GroupNorm normalizes over channel groups within
each sample, making it independent of batch composition.
"""

import torch.nn as nn
import torchvision.models as models


def resnet18_gn(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Replace all BatchNorm2d layers with GroupNorm
    _replace_bn_with_gn(model)
    return model


def _replace_bn_with_gn(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = min(32, num_channels)
            # Ensure num_channels is divisible by num_groups
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child)
