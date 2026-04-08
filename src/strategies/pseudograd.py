"""
FedPseudoGradSim — Layer-wise + delta-similarity weighting (no server Adam).

This is the ablation that isolates "similarity weighting alone" from
"server momentum alone". Compare against:
- FedPartAdam      → server Adam alone
- FedPartSAM-Δ     → server Adam + delta similarity
- FedPartSAM-m     → server Adam + momentum similarity
"""

from typing import Dict, List

import numpy as np

from .base import PartialStrategy


class FedPseudoGradSimStrategy(PartialStrategy):

    def __init__(self, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self._reference: Dict[int, List[np.ndarray]] = {}
        self.weight_history: List[dict] = []

    def _aggregate_round(self, server_round, client_arrays, client_metrics, results):
        active = self.active_layer_for_round(server_round)
        base = self.global_arrays if active == -1 else self._slice_global(active)

        client_deltas = [
            [arrays[j] - base[j] for j in range(len(base))] for arrays in client_arrays
        ]

        weights = self._similarity_weights(client_deltas, active)
        self.weight_history.append({
            "round": server_round, "layer": active, "weights": weights.tolist(),
        })

        n_arrays = len(base)
        agg_delta = [np.zeros_like(base[j]) for j in range(n_arrays)]
        for w, deltas in zip(weights, client_deltas):
            for j in range(n_arrays):
                agg_delta[j] += w * deltas[j]

        # Apply update directly (no server Adam)
        updated = [base[j] + agg_delta[j] for j in range(n_arrays)]

        # Store reference for next round
        self._reference[active] = [a.copy() for a in agg_delta]

        return self._merge_layer_update(active, updated)

    def _similarity_weights(self, client_deltas, active):
        n = len(client_deltas)
        ref = self._reference.get(active)
        if ref is None or self.tau >= 1e6:
            return np.full(n, 1.0 / n)

        ref_flat = np.concatenate([r.flatten() for r in ref])
        ref_norm = float(np.linalg.norm(ref_flat))
        if ref_norm < 1e-10:
            return np.full(n, 1.0 / n)

        sims = np.zeros(n)
        for i, deltas in enumerate(client_deltas):
            d_flat = np.concatenate([a.flatten() for a in deltas])
            if d_flat.shape != ref_flat.shape:
                sims[i] = 0.0
                continue
            d_norm = float(np.linalg.norm(d_flat))
            sims[i] = 0.0 if d_norm < 1e-10 else float(np.dot(d_flat, ref_flat) / (d_norm * ref_norm))

        scaled = sims / self.tau
        scaled -= scaled.max()
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum()

    def get_results(self):
        r = super().get_results()
        r["weight_history"] = list(self.weight_history)
        r["tau"] = self.tau
        return r
