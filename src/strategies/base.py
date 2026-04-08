"""
Base classes for federated strategies.

`BaseStrategy` extends Flower's FedAvg with:
- Per-round result tracking (accuracy, communication cost, similarity weights)
- Centralized evaluation only (no client-side eval — simpler, more reliable)
- FedProx support via `proximal_mu > 0` (no separate subclass needed)
- A clean `aggregate_fit` interface that subclasses override

`PartialStrategy` extends `BaseStrategy` for layer-wise (FedPart-style) training.
It manages the layer training schedule and sends only the active layer's parameters
to clients.

Key design choices:
1. **Server stores its own copy of the global model** as a `List[np.ndarray]`. This
   avoids re-encoding/decoding `Parameters` objects every round.
2. **Communication cost is measured per client per round** using actual byte counts
   (`numpy_bytes`), not Python object overhead.
3. **Subclasses implement `_aggregate_round`** which receives client parameters and
   returns the new global model. The boilerplate (delta computation, communication
   tracking, metric logging) lives here.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..comm import numpy_bytes, parameters_bytes, serialized_bytes
from ..layers import LayerSpec, build_layer_spec
from ..params import get_parameters


# Special key clients can return in metrics for additional comm cost (e.g. momentum)
EXTRA_COMM_KEYS = ("momentum_state",)


class BaseStrategy(FedAvg):
    """Base for all full-model FL strategies in this codebase."""

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        evaluate_fn: Callable,
        proximal_mu: float = 0.0,
        model_name: str = "cnn",  # accepted for symmetry; only PartialStrategy uses it
        **kwargs,
    ):
        # Drop layer-wise kwargs that don't apply to full-model strategies
        kwargs.pop("warmup_rounds", None)
        kwargs.pop("rounds_per_layer", None)
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self._model_fn = model_fn
        self.model_name = model_name
        self.proximal_mu = proximal_mu

        # Server-side state
        self.global_arrays: Optional[List[np.ndarray]] = None  # the canonical global model
        self.global_bytes: int = 0  # bytes-of-global-model (cached)

        # Per-round results: round -> dict of metrics
        self.round_results: Dict[int, dict] = defaultdict(dict)
        # Per-round eval results (loss + accuracy)
        self.eval_results: Dict[int, dict] = {}

    # ─── Initialization ──────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        model = self._model_fn()
        self.global_arrays = get_parameters(model)
        self.global_bytes = numpy_bytes(self.global_arrays)
        return ndarrays_to_parameters(self.global_arrays)

    # ─── Centralized evaluation (simpler than client-side) ───────────────

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        result = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if result is None:
            return None
        loss, metrics = result
        self.eval_results[server_round] = {
            "loss": float(loss),
            "accuracy": float(metrics.get("accuracy", 0.0)),
        }
        return loss, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []  # centralized eval only — skip client eval entirely

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    # ─── Fit configuration ───────────────────────────────────────────────

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = self._build_fit_config(server_round)
        sample_size, min_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_clients)
        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]

    def _build_fit_config(self, server_round: int) -> dict:
        """Build the per-round config dict sent to clients. Subclasses override to add fields."""
        config = {"round": server_round}
        if self.proximal_mu > 0.0:
            config["proximal_mu"] = self.proximal_mu
        return config

    # ─── Aggregation ─────────────────────────────────────────────────────

    def aggregate_fit(
        self, server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # ── Track communication cost (down + up) ──
        # Down = global params sent to each client
        # Up = client params/momentum sent back
        down_bytes = self.global_bytes * len(results)
        up_bytes = 0
        for _, fr in results:
            up_bytes += parameters_bytes(fr.parameters)
            for key in EXTRA_COMM_KEYS:
                up_bytes += serialized_bytes(fr.metrics.get(key))
        total_bytes = down_bytes + up_bytes

        # ── Subclass-specific aggregation ──
        client_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        client_metrics = [fr.metrics for _, fr in results]
        new_global = self._aggregate_round(server_round, client_arrays, client_metrics, results)

        # ── Update server state ──
        self.global_arrays = new_global
        self.global_bytes = numpy_bytes(new_global)

        # ── Record metrics ──
        self.round_results[server_round].update({
            "down_bytes": down_bytes,
            "up_bytes": up_bytes,
            "total_bytes": total_bytes,
            "num_clients": len(results),
        })

        return ndarrays_to_parameters(new_global), {}

    def _aggregate_round(
        self,
        server_round: int,
        client_arrays: List[List[np.ndarray]],
        client_metrics: List[dict],
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> List[np.ndarray]:
        """Subclasses override. Default: simple weighted (by num_examples) average."""
        total_examples = sum(fr.num_examples for _, fr in results)
        new_global = [np.zeros_like(g) for g in self.global_arrays]
        for arrays, (_, fr) in zip(client_arrays, results):
            w = fr.num_examples / total_examples
            for i, a in enumerate(arrays):
                new_global[i] += w * a
        return new_global

    # ─── Results export ──────────────────────────────────────────────────

    def get_results(self) -> dict:
        """Return all data needed for plotting / paper tables."""
        return {
            "round_results": dict(self.round_results),
            "eval_results": dict(self.eval_results),
        }


class PartialStrategy(BaseStrategy):
    """Layer-wise partial training base class.

    Uses a `LayerSpec` to map logical layer indices → state_dict keys, so it
    works with both the simple (weight, bias)-pair CNN and the grouped-block
    ResNet-18 (where one logical layer = a residual block group).

    Manages the layer training sequence:
        rounds 1..warmup_rounds            → -1 (full model)
        then cycle layer 0, 1, 2, ... each for `rounds_per_layer` rounds
    """

    def __init__(
        self,
        model_name: str = "cnn",
        warmup_rounds: int = 2,
        rounds_per_layer: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.warmup_rounds = warmup_rounds
        self.rounds_per_layer = rounds_per_layer
        self.layer_spec: Optional[LayerSpec] = None
        # Cache: state_dict_key → flat-array index (built once at init)
        self._key_to_idx: dict = {}
        # Cache: layer_idx → list of flat-array indices
        self._layer_indices: dict = {}

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        # Build the LayerSpec from the model
        model = self._model_fn()
        self.layer_spec = build_layer_spec(model, self.model_name)
        # Build flat-index lookups so we can slice global_arrays by layer
        keys = list(model.state_dict().keys())
        self._key_to_idx = {k: i for i, k in enumerate(keys)}
        self._layer_indices = {
            li: [self._key_to_idx[k] for k in self.layer_spec.keys_for_layer(li)]
            for li in range(self.layer_spec.num_layers)
        }
        return params

    @property
    def num_layers(self) -> int:
        return self.layer_spec.num_layers

    def active_layer_for_round(self, server_round: int) -> int:
        """Return the layer index to train this round, or -1 for full-model."""
        if server_round <= self.warmup_rounds:
            return -1
        offset = server_round - self.warmup_rounds - 1
        return (offset // self.rounds_per_layer) % self.num_layers

    def _slice_global(self, layer_idx: int) -> List[np.ndarray]:
        """Return the global model arrays for the given logical layer."""
        return [self.global_arrays[i] for i in self._layer_indices[layer_idx]]

    def _build_fit_config(self, server_round: int) -> dict:
        config = super()._build_fit_config(server_round)
        config["active_layer"] = self.active_layer_for_round(server_round)
        return config

    def configure_fit(self, server_round, parameters, client_manager):
        """Send only the active layer's params (saves bandwidth)."""
        active = self.active_layer_for_round(server_round)
        if active == -1:
            sent_arrays = self.global_arrays
        else:
            sent_arrays = self._slice_global(active)
        sent_params = ndarrays_to_parameters(sent_arrays)

        config = self._build_fit_config(server_round)
        sample_size, min_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_clients)
        fit_ins = FitIns(sent_params, config)
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        """Communication tracking + delegate to subclass `_aggregate_round`."""
        if not results:
            return None, {}

        active = self.active_layer_for_round(server_round)
        sent_arrays = self.global_arrays if active == -1 else self._slice_global(active)
        layer_bytes = numpy_bytes(sent_arrays)
        down_bytes = layer_bytes * len(results)

        up_bytes = 0
        for _, fr in results:
            up_bytes += parameters_bytes(fr.parameters)
            for key in EXTRA_COMM_KEYS:
                up_bytes += serialized_bytes(fr.metrics.get(key))
        total_bytes = down_bytes + up_bytes

        client_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        client_metrics = [fr.metrics for _, fr in results]
        new_global = self._aggregate_round(server_round, client_arrays, client_metrics, results)

        self.global_arrays = new_global
        self.global_bytes = numpy_bytes(new_global)

        self.round_results[server_round].update({
            "down_bytes": down_bytes,
            "up_bytes": up_bytes,
            "total_bytes": total_bytes,
            "num_clients": len(results),
            "active_layer": active,
        })

        return ndarrays_to_parameters(new_global), {}

    def _merge_layer_update(
        self, active_layer: int, layer_arrays: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Apply a partial-layer aggregated update to a copy of the global model."""
        if active_layer == -1:
            return [np.copy(a) for a in layer_arrays]
        new_global = list(self.global_arrays)
        for arr_idx, flat_idx in enumerate(self._layer_indices[active_layer]):
            new_global[flat_idx] = layer_arrays[arr_idx]
        return new_global
