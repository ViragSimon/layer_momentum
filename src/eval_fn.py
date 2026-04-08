"""Centralized evaluation function used by Flower strategies via `evaluate_fn`."""

import copy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import NDArrays, Scalar

from .device import DEVICE
from .params import set_parameters
from .train import evaluate


def make_evaluate_fn(
    test_loader: DataLoader,
    model: nn.Module,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Create a centralized evaluation function for a Flower strategy.

    The returned function evaluates the global model on `test_loader` after each round.
    Uses a deep copy of the template model so that running it does not mutate the original.
    """
    def _evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model_copy = copy.deepcopy(model).to(DEVICE)
        set_parameters(model_copy, parameters)
        loss, accuracy = evaluate(model_copy, test_loader)
        print(f"  [round {server_round}] test loss={loss:.4f} acc={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return _evaluate
