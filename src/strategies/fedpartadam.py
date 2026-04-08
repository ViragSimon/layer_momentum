"""
FedPartAdam — Layer-wise partial training + server-side Adam (uniform client weights).

This is the **ablation** showing what server-side momentum alone contributes
(without similarity weighting). FedPartSAM with τ→∞ would give the same behaviour.
"""

from typing import Dict, List

import numpy as np

from .base import PartialStrategy


class FedPartAdamStrategy(PartialStrategy):

    def __init__(
        self,
        server_lr: float = 0.01,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.server_beta1 = server_beta1
        self.server_beta2 = server_beta2
        self.server_eps = server_eps
        # Per-layer Adam state (keyed by layer index, or "full" for warmup rounds)
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}

    def _aggregate_round(self, server_round, client_arrays, client_metrics, results):
        active = self.active_layer_for_round(server_round)
        total_examples = sum(fr.num_examples for _, fr in results)

        base = self.global_arrays if active == -1 else self._slice_global(active)
        n_arrays = len(base)

        agg_delta = [np.zeros_like(base[j]) for j in range(n_arrays)]
        for arrays, (_, fr) in zip(client_arrays, results):
            w = fr.num_examples / total_examples
            for j in range(n_arrays):
                agg_delta[j] += w * (arrays[j] - base[j])

        key = "full" if active == -1 else active
        updated = self._adam_step(key, agg_delta, base)
        return self._merge_layer_update(active, updated)

    def _adam_step(self, key, deltas, base):
        """Apply Adam to each array in `deltas`, returning updated parameters."""
        if key not in self._m:
            self._m[key] = [np.zeros_like(d) for d in deltas]
            self._v[key] = [np.zeros_like(d) for d in deltas]

        out = []
        for i, d in enumerate(deltas):
            self._m[key][i] = self.server_beta1 * self._m[key][i] + (1 - self.server_beta1) * d
            self._v[key][i] = self.server_beta2 * self._v[key][i] + (1 - self.server_beta2) * (d ** 2)
            update = self.server_lr * self._m[key][i] / (np.sqrt(self._v[key][i]) + self.server_eps)
            out.append(base[i] + update)
        return out
