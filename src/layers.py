"""
Logical layer groupings for partial training.

A "logical layer" is a group of model parameters that move together during
layer-wise training. For the custom CNN, each (weight, bias) pair is its own
logical layer (8 total). For ResNet-18, we group residual blocks together
(conv1+gn1, layer1, layer2, layer3, layer4, fc — 6 logical layers).

The `LayerSpec` interface lets strategies and clients work with arbitrary
groupings without hardcoding model structure.
"""

from typing import List

import torch.nn as nn


class LayerSpec:
    """Maps logical layer indices to lists of state_dict keys (parameter names)."""

    def __init__(self, groups: List[List[str]]):
        """
        Parameters
        ----------
        groups : list of list of str
            `groups[i]` is the list of state_dict keys belonging to logical layer i.
        """
        self.groups = groups

    @property
    def num_layers(self) -> int:
        return len(self.groups)

    def keys_for_layer(self, layer_idx: int) -> List[str]:
        return self.groups[layer_idx]

    def all_keys(self) -> List[str]:
        return [k for g in self.groups for k in g]


def cnn_layer_spec(model: nn.Module) -> LayerSpec:
    """Build a (weight, bias) per-layer spec from any sequential model.

    Each logical layer = one (weight, bias) pair from the model's state_dict.
    Works for the custom CNN (8 layers).
    """
    keys = list(model.state_dict().keys())
    groups = []
    i = 0
    while i < len(keys):
        if i + 1 < len(keys) and keys[i + 1].endswith(".bias"):
            groups.append([keys[i], keys[i + 1]])
            i += 2
        else:
            groups.append([keys[i]])
            i += 1
    return LayerSpec(groups)


def resnet18_layer_spec(model: nn.Module) -> LayerSpec:
    """Build a 6-logical-layer spec for ResNet-18 with GroupNorm.

    Logical layers:
        0: conv1 + bn1/gn1                  (~10K params)
        1: layer1 (2 BasicBlocks)           (~150K)
        2: layer2 (2 BasicBlocks)           (~525K)
        3: layer3 (2 BasicBlocks)           (~2.1M)
        4: layer4 (2 BasicBlocks)           (~8.4M)
        5: fc                               (~50K)
    """
    keys = list(model.state_dict().keys())

    def keys_starting_with(prefix: str) -> List[str]:
        return [k for k in keys if k.startswith(prefix)]

    groups = [
        # Layer 0: conv1 + first norm
        keys_starting_with("conv1.") + keys_starting_with("bn1.") + keys_starting_with("gn1."),
        keys_starting_with("layer1."),
        keys_starting_with("layer2."),
        keys_starting_with("layer3."),
        keys_starting_with("layer4."),
        keys_starting_with("fc."),
    ]
    # Filter empty groups (in case the model uses different naming)
    groups = [g for g in groups if g]
    return LayerSpec(groups)


def build_layer_spec(model: nn.Module, model_name: str) -> LayerSpec:
    """Build the appropriate LayerSpec for a given model."""
    if model_name == "cnn":
        return cnn_layer_spec(model)
    if model_name == "resnet18":
        return resnet18_layer_spec(model)
    raise ValueError(f"No layer spec for model: {model_name}")
