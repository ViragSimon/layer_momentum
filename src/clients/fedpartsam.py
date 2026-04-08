"""
FedPartSAM-m client.

Design (simpler than the original plan):
- Each client maintains its own *persistent* Adam optimizer state across rounds.
- The client's Adam state evolves naturally — on each round, the client
  reconstructs its Adam optimizer from the persisted state, trains for E epochs,
  and saves the updated state.
- After training, the client sends its current `exp_avg` (m) for the active layer
  back to the server.
- The server uses the m vectors *only as a similarity signal* for weighting
  client contributions. The server does NOT send m back to clients to load.

This avoids the complications of patching Adam optimizer state mid-training
(bias correction, exp_avg_sq divergence, shape mismatches across rounds) while
still providing the headline contribution: cosine-similarity weighting on Adam
first moments rather than parameter deltas.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import Context

from ..device import DEVICE
from ..layers import LayerSpec, build_layer_spec
from ..params import (
    get_parameters, set_parameters, get_layer_parameters,
    set_layer_parameters, freeze_all_but,
)
from ..serialize import encode_arrays
from ..state import ClientStateStore
from ..train import train_local


class FedPartSAMClient(NumPyClient):
    """Client for FedPartSAM-m: persistent local Adam, momentum upload only."""

    MODEL_KEY = "full_model"
    M_KEY = "adam_m"
    V_KEY = "adam_v"

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

        # Map state_dict keys → flat-array index, and per-layer index lists
        all_keys = list(model.state_dict().keys())
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        self._layer_indices = {
            li: [key_to_idx[k] for k in self.spec.keys_for_layer(li)]
            for li in range(self.spec.num_layers)
        }

        self.store = ClientStateStore(context)
        if not self.store.has(self.MODEL_KEY):
            self.store.save(self.MODEL_KEY, get_parameters(self.model))

    # ── Persistent state ──────────────────────────────────────────────────

    def _load_full_model(self):
        arrays = self.store.load(self.MODEL_KEY)
        if arrays is not None:
            set_parameters(self.model, arrays)

    def _save_full_model(self):
        self.store.save(self.MODEL_KEY, get_parameters(self.model))

    def _load_adam_state(self) -> tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        return self.store.load(self.M_KEY), self.store.load(self.V_KEY)

    def _save_adam_state(self, m: List[np.ndarray], v: List[np.ndarray]):
        self.store.save(self.M_KEY, m)
        self.store.save(self.V_KEY, v)

    # ── Optimizer build/extract ───────────────────────────────────────────

    def _build_optimizer(
        self,
        full_m: Optional[List[np.ndarray]],
        full_v: Optional[List[np.ndarray]],
    ) -> torch.optim.Optimizer:
        """Build a fresh Adam, populate state from persistent (m, v) for trainable params.

        `full_m` and `full_v` are full-model lists (16 arrays for our CNN).
        We map each trainable param to its position in the full list and load
        state only for that param. Untrainable params get no state (Adam ignores them).
        """
        # Build a list of (full_index, param) for params still requires_grad
        # Then create a list of trainable params for Adam
        all_params = list(self.model.parameters())
        trainable = []
        full_indices = []
        for idx, p in enumerate(all_params):
            if p.requires_grad:
                trainable.append(p)
                full_indices.append(idx)

        optimizer = torch.optim.Adam(trainable, lr=self.local_lr)

        # Populate state from persisted (m, v) if available and shapes match
        if full_m is not None and full_v is not None:
            for p, full_idx in zip(trainable, full_indices):
                if full_idx >= len(full_m) or full_idx >= len(full_v):
                    continue
                m_arr = full_m[full_idx]
                v_arr = full_v[full_idx]
                if m_arr.shape != tuple(p.data.shape) or v_arr.shape != tuple(p.data.shape):
                    continue
                state = optimizer.state.setdefault(p, {})
                # Use a large step to disable bias correction (these moments
                # are already accumulated, not first-step values)
                state["step"] = torch.tensor(1000.0)
                state["exp_avg"] = torch.tensor(m_arr, device=p.device, dtype=p.dtype)
                state["exp_avg_sq"] = torch.tensor(v_arr, device=p.device, dtype=p.dtype)

        return optimizer

    def _extract_adam_state(self, optimizer) -> tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Pull (m, v, full_indices) from optimizer state for trainable params."""
        m_list, v_list = [], []
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state and "exp_avg_sq" in state:
                    m_list.append(state["exp_avg"].detach().cpu().numpy())
                    v_list.append(state["exp_avg_sq"].detach().cpu().numpy())
                else:
                    m_list.append(np.zeros_like(p.data.detach().cpu().numpy()))
                    v_list.append(np.zeros_like(p.data.detach().cpu().numpy()))
        return m_list, v_list

    # ── Flower API ────────────────────────────────────────────────────────

    def get_parameters(self, config):
        active = config.get("active_layer", -1)
        if active == -1:
            return get_parameters(self.model)
        return get_layer_parameters(self.model, self.spec, active)

    def fit(self, parameters, config):
        # 1. Restore full model from persistent state
        self._load_full_model()

        # 2. Apply server's parameter update for the active layer
        active = config.get("active_layer", -1)
        if active == -1:
            set_parameters(self.model, parameters)
        else:
            set_layer_parameters(self.model, self.spec, active, parameters)

        # 3. Freeze all but the active layer
        freeze_all_but(self.model, self.spec, active)

        # 4. Build optimizer from persistent Adam state (full m, v)
        full_m, full_v = self._load_adam_state()
        optimizer = self._build_optimizer(full_m, full_v)

        # 5. Train locally
        train_local(
            self.model, self.train_loader, self.num_epochs,
            optimizer=optimizer, lr=self.local_lr,
        )

        # 6. Extract updated Adam state for the trainable params
        new_m_partial, new_v_partial = self._extract_adam_state(optimizer)

        # 7. Persist full model and updated Adam state.
        #    Update only the entries corresponding to trainable params.
        self._save_full_model()

        if full_m is None or full_v is None:
            full_m = [np.zeros_like(p) for p in get_parameters(self.model)]
            full_v = [np.zeros_like(p) for p in get_parameters(self.model)]

        all_params = list(self.model.parameters())
        partial_idx = 0
        for full_idx, p in enumerate(all_params):
            if p.requires_grad and partial_idx < len(new_m_partial):
                full_m[full_idx] = new_m_partial[partial_idx]
                full_v[full_idx] = new_v_partial[partial_idx]
                partial_idx += 1
        self._save_adam_state(full_m, full_v)

        # 8. Encode the m for the active layer (or full m if active==-1) for the server
        if active == -1:
            m_to_send = full_m
        else:
            m_to_send = [full_m[i] for i in self._layer_indices[active]]

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"momentum_state": encode_arrays(m_to_send)},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}
