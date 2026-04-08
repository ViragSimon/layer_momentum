"""
FedAdam — full-model FedAvg + server-side Adam (Reddi et al., ICLR 2021).

Server treats the aggregated client delta as a pseudo-gradient and applies
Adam to update the global model. Same client logic as FedAvg.
"""

from typing import List

import numpy as np

from .base import BaseStrategy


class FedAdamStrategy(BaseStrategy):

    def __init__(
        self,
        server_lr: float = 0.01,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_eps: float = 1e-3,  # Reddi 2021 uses 1e-3, NOT 1e-8
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.server_beta1 = server_beta1
        self.server_beta2 = server_beta2
        self.server_eps = server_eps
        self._m: List[np.ndarray] = []
        self._v: List[np.ndarray] = []

    def _aggregate_round(self, server_round, client_arrays, client_metrics, results):
        # 1. Compute weighted average of client deltas (pseudo-gradient)
        total_examples = sum(fr.num_examples for _, fr in results)
        agg_delta = [np.zeros_like(g) for g in self.global_arrays]
        for arrays, (_, fr) in zip(client_arrays, results):
            w = fr.num_examples / total_examples
            for i, a in enumerate(arrays):
                agg_delta[i] += w * (a - self.global_arrays[i])

        # 2. Initialize Adam state on first call
        if not self._m:
            self._m = [np.zeros_like(g) for g in self.global_arrays]
            self._v = [np.zeros_like(g) for g in self.global_arrays]

        # 3. Adam update
        new_global = []
        for i, d in enumerate(agg_delta):
            self._m[i] = self.server_beta1 * self._m[i] + (1 - self.server_beta1) * d
            self._v[i] = self.server_beta2 * self._v[i] + (1 - self.server_beta2) * (d ** 2)
            update = self.server_lr * self._m[i] / (np.sqrt(self._v[i]) + self.server_eps)
            new_global.append(self.global_arrays[i] + update)
        return new_global
