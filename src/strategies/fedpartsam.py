"""
FedPartSAM — Similarity-Aware Momentum for partial network updates.

The headline strategy of this project. Combines:
1. Layer-wise partial training (FedPart)
2. Server-side Adam (FedOpt)
3. Cosine-similarity weighted aggregation (softmax with temperature τ)

Two flavours, controlled by `similarity_source`:
- "momentum"  — similarity computed on clients' Adam first-moment m
                (the headline contribution; sends m alongside parameters)
- "delta"     — similarity computed on parameter deltas
                (an ablation; no extra communication, but a noisier signal)

The reference vector for similarity is the *previous round's aggregated value*
for the same active layer. This avoids self-referential bias.
"""

from typing import Dict, List, Optional

import numpy as np

from .base import PartialStrategy
from ..serialize import encode_arrays, decode_arrays


class FedPartSAMStrategy(PartialStrategy):

    def __init__(
        self,
        similarity_source: str = "momentum",  # "momentum" or "delta"
        tau: float = 1.0,
        server_lr: float = 0.01,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_eps: float = 1e-3,
        layer_selection: str = "sequential",  # "sequential" or "adaptive"
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert similarity_source in ("momentum", "delta")
        self.similarity_source = similarity_source
        self.tau = tau
        self.server_lr = server_lr
        self.server_beta1 = server_beta1
        self.server_beta2 = server_beta2
        self.server_eps = server_eps
        self.layer_selection = layer_selection

        # Per-layer server Adam state
        self._m_server: Dict[int, List[np.ndarray]] = {}
        self._v_server: Dict[int, List[np.ndarray]] = {}

        # Per-layer reference for similarity
        # For "momentum": last round's aggregated client momentum for each layer
        # For "delta": last round's aggregated client delta for each layer
        self._reference: Dict[int, List[np.ndarray]] = {}

        # Per-round weight log (for analysis / plots)
        self.weight_history: List[dict] = []

    # ── Adaptive layer selection ──────────────────────────────────────────

    def active_layer_for_round(self, server_round: int) -> int:
        if self.layer_selection != "adaptive":
            return super().active_layer_for_round(server_round)
        if server_round <= self.warmup_rounds or not self._m_server:
            return super().active_layer_for_round(server_round)
        # Pick the layer with the largest server momentum norm
        best_layer = -1
        best_norm = -1.0
        for layer_idx, m_arrays in self._m_server.items():
            if layer_idx == "full":
                continue
            norm = sum(float(np.linalg.norm(a)) for a in m_arrays)
            if norm > best_norm:
                best_norm = norm
                best_layer = layer_idx
        return best_layer if best_layer >= 0 else super().active_layer_for_round(server_round)

    # ── configure_fit override: send momentum reference to clients ────────

    def configure_fit(self, server_round, parameters, client_manager):
        fit_configs = super().configure_fit(server_round, parameters, client_manager)
        if self.similarity_source != "momentum":
            return fit_configs

        # Send the aggregated momentum reference for this layer to all clients
        active = self.active_layer_for_round(server_round)
        ref = self._reference.get(active)
        if ref is not None:
            from flwr.common import FitIns
            encoded = encode_arrays(ref)
            new_configs = []
            for client, fi in fit_configs:
                cfg = {**fi.config, "global_momentum": encoded}
                new_configs.append((client, FitIns(fi.parameters, cfg)))
            return new_configs
        return fit_configs

    # ── Aggregation ───────────────────────────────────────────────────────

    def _aggregate_round(self, server_round, client_arrays, client_metrics, results):
        active = self.active_layer_for_round(server_round)

        # 1. Determine the base (global slice corresponding to this round)
        base = self.global_arrays if active == -1 else self._slice_global(active)

        # 2. Compute per-client deltas
        client_deltas = [
            [arrays[j] - base[j] for j in range(len(base))]
            for arrays in client_arrays
        ]

        # 3. Determine the similarity-source vector for each client
        if self.similarity_source == "momentum":
            client_signals = [decode_arrays(m.get("momentum_state")) for m in client_metrics]
            # Filter out clients that didn't return momentum (shouldn't happen but be defensive)
            if any(len(s) == 0 for s in client_signals):
                client_signals = client_deltas  # fallback to delta similarity
        else:  # delta
            client_signals = client_deltas

        # 4. Compute similarity weights
        weights = self._similarity_weights(client_signals, active)
        self.weight_history.append({
            "round": server_round,
            "layer": active,
            "weights": weights.tolist(),
        })

        # 5. Weighted average of client deltas
        n_arrays = len(base)
        agg_delta = [np.zeros_like(base[j]) for j in range(n_arrays)]
        for w, deltas in zip(weights, client_deltas):
            for j in range(n_arrays):
                agg_delta[j] += w * deltas[j]

        # 6. Server-side Adam update
        updated = self._adam_step(active, agg_delta, base)

        # 7. Store reference for next round (mean of client signals)
        if self.similarity_source == "momentum":
            ref_arrays = [
                np.mean([s[j] for s in client_signals], axis=0)
                for j in range(len(client_signals[0]))
            ]
            self._reference[active] = ref_arrays
        else:  # delta
            self._reference[active] = [a.copy() for a in agg_delta]

        return self._merge_layer_update(active, updated)

    # ── Similarity computation ────────────────────────────────────────────

    def _similarity_weights(self, client_signals: List[List[np.ndarray]], active: int) -> np.ndarray:
        """Softmax-with-temperature weights from cosine similarity to the layer's reference."""
        n = len(client_signals)
        ref = self._reference.get(active)

        if ref is None or self.tau >= 1e6:
            # No reference yet OR effectively-infinite temperature → uniform
            return np.full(n, 1.0 / n)

        ref_flat = np.concatenate([r.flatten() for r in ref])
        ref_norm = float(np.linalg.norm(ref_flat))
        if ref_norm < 1e-10:
            return np.full(n, 1.0 / n)

        sims = np.zeros(n)
        for i, s in enumerate(client_signals):
            s_flat = np.concatenate([a.flatten() for a in s])
            if s_flat.shape != ref_flat.shape:
                sims[i] = 0.0
                continue
            s_norm = float(np.linalg.norm(s_flat))
            if s_norm < 1e-10:
                sims[i] = 0.0
            else:
                sims[i] = float(np.dot(s_flat, ref_flat) / (s_norm * ref_norm))

        # Softmax with temperature
        scaled = sims / self.tau
        scaled -= scaled.max()  # numerical stability
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum()

    # ── Server-side Adam step (per-layer) ─────────────────────────────────

    def _adam_step(self, active, deltas, base) -> List[np.ndarray]:
        key = "full" if active == -1 else active
        if key not in self._m_server:
            self._m_server[key] = [np.zeros_like(d) for d in deltas]
            self._v_server[key] = [np.zeros_like(d) for d in deltas]

        out = []
        for i, d in enumerate(deltas):
            self._m_server[key][i] = self.server_beta1 * self._m_server[key][i] + (1 - self.server_beta1) * d
            self._v_server[key][i] = self.server_beta2 * self._v_server[key][i] + (1 - self.server_beta2) * (d ** 2)
            update = self.server_lr * self._m_server[key][i] / (np.sqrt(self._v_server[key][i]) + self.server_eps)
            out.append(base[i] + update)
        return out

    # ── Results export ────────────────────────────────────────────────────

    def get_results(self) -> dict:
        results = super().get_results()
        results["weight_history"] = list(self.weight_history)
        results["similarity_source"] = self.similarity_source
        results["tau"] = self.tau
        return results
