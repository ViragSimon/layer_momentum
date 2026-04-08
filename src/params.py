"""
Model parameter utilities — get/set parameters, freeze layers.

A "logical layer" can span multiple state_dict entries (see `layers.py`).
Operations like `set_layer_parameters` and `freeze_all_but` accept a `LayerSpec`
that maps logical layer indices to lists of parameter names.
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .layers import LayerSpec


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract all model parameters as a list of CPU numpy arrays."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a full set of parameters into the model."""
    state_dict = OrderedDict(
        (k, torch.from_numpy(np.asarray(v)))
        for k, v in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state_dict, strict=True)


def get_layer_parameters(
    model: nn.Module, spec: LayerSpec, layer_idx: int,
) -> List[np.ndarray]:
    """Extract just the parameters belonging to logical layer `layer_idx`."""
    keys = spec.keys_for_layer(layer_idx)
    state = model.state_dict()
    return [state[k].detach().cpu().numpy() for k in keys]


def set_layer_parameters(
    model: nn.Module, spec: LayerSpec, layer_idx: int, arrays: List[np.ndarray],
) -> None:
    """Update only the parameters belonging to logical layer `layer_idx`."""
    keys = spec.keys_for_layer(layer_idx)
    if len(keys) != len(arrays):
        raise ValueError(
            f"Layer {layer_idx} has {len(keys)} keys but received {len(arrays)} arrays"
        )
    state = model.state_dict()
    for k, a in zip(keys, arrays):
        state[k] = torch.from_numpy(np.asarray(a))
    model.load_state_dict(state, strict=True)


def freeze_all_but(model: nn.Module, spec: LayerSpec, layer_idx: int) -> None:
    """Freeze all parameters except those in logical layer `layer_idx`.

    Pass `layer_idx == -1` to unfreeze everything (full-model training).
    """
    if layer_idx == -1:
        for p in model.parameters():
            p.requires_grad = True
        return

    target_keys = set(spec.keys_for_layer(layer_idx))
    # Map state_dict keys to parameters via named_parameters (which only includes
    # learnable params — buffers like BN running stats are excluded automatically).
    learnable_keys = dict(model.named_parameters())
    for name, p in learnable_keys.items():
        p.requires_grad = (name in target_keys)
