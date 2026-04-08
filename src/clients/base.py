"""
Base FL client classes.

`FullModelClient`  — for FedAvg, FedProx, FedAdam (clients always train all params)
`PartialClient`    — for FedPart family (clients train only the active layer)

Both auto-detect FedProx mode from the `proximal_mu` config field.
"""

import copy
from typing import List

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import Context

from ..comm import numpy_bytes
from ..device import DEVICE
from ..layers import LayerSpec, build_layer_spec
from ..params import (
    get_parameters, set_parameters, get_layer_parameters,
    set_layer_parameters, freeze_all_but,
)
from ..state import ClientStateStore
from ..train import train_local


class FullModelClient(NumPyClient):
    """Trains all parameters every round (no layer freezing).

    Used by FedAvg, FedProx, FedAdam.
    """

    def __init__(
        self, partition_id: int, model, train_loader, val_loader,
        num_epochs: int, local_lr: float = 1e-3,
    ):
        self.partition_id = partition_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.local_lr = local_lr

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Load global parameters
        set_parameters(self.model, parameters)

        # Train (with optional FedProx term)
        proximal_mu = float(config.get("proximal_mu", 0.0))
        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]
        train_local(
            self.model, self.train_loader, self.num_epochs,
            lr=self.local_lr, proximal_mu=proximal_mu, global_params=global_params,
        )

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}  # client-side eval disabled — server uses centralized eval


class PartialClient(NumPyClient):
    """Layer-wise client. Uses a `LayerSpec` to address logical layers.

    Receives partial params (matching the active logical layer), freezes
    all but that layer, trains, and returns the partial params.

    Persists the full local model state across rounds via `context.state`
    so that frozen-layer parameters survive between fit() calls (Flower
    creates a new client instance per call).
    """

    MODEL_KEY = "full_model"

    def __init__(
        self, partition_id: int, model, train_loader, val_loader,
        num_epochs: int, local_lr: float, context: Context, model_name: str = "cnn",
    ):
        self.partition_id = partition_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.local_lr = local_lr
        self.spec: LayerSpec = build_layer_spec(model, model_name)

        self.store = ClientStateStore(context)
        if not self.store.has(self.MODEL_KEY):
            self.store.save(self.MODEL_KEY, get_parameters(self.model))

    def _load_persistent_state(self):
        arrays = self.store.load(self.MODEL_KEY)
        if arrays is not None:
            set_parameters(self.model, arrays)

    def _save_persistent_state(self):
        self.store.save(self.MODEL_KEY, get_parameters(self.model))

    def get_parameters(self, config):
        active = config.get("active_layer", -1)
        if active == -1:
            return get_parameters(self.model)
        return get_layer_parameters(self.model, self.spec, active)

    def fit(self, parameters, config):
        # 1. Restore full model
        self._load_persistent_state()

        # 2. Apply server update for the active layer
        active = config.get("active_layer", -1)
        if active == -1:
            set_parameters(self.model, parameters)
        else:
            set_layer_parameters(self.model, self.spec, active, parameters)

        # 3. Freeze all but the active layer
        freeze_all_but(self.model, self.spec, active)

        # 4. Train locally
        proximal_mu = float(config.get("proximal_mu", 0.0))
        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]
        train_local(
            self.model, self.train_loader, self.num_epochs,
            lr=self.local_lr, proximal_mu=proximal_mu, global_params=global_params,
        )

        # 5. Persist + return
        self._save_persistent_state()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}
