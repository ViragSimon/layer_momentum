"""FedPartAvg — layer-wise partial training with uniform (FedAvg-style) aggregation."""

from typing import List

import numpy as np

from .base import PartialStrategy


class FedPartAvgStrategy(PartialStrategy):

    def _aggregate_round(self, server_round, client_arrays, client_metrics, results):
        active = self.active_layer_for_round(server_round)
        total_examples = sum(fr.num_examples for _, fr in results)

        # Each client_arrays[i] is either the full model (if active=-1) or [w, b] for the active layer
        n_arrays = len(client_arrays[0])
        agg = [np.zeros_like(client_arrays[0][j]) for j in range(n_arrays)]
        for arrays, (_, fr) in zip(client_arrays, results):
            w = fr.num_examples / total_examples
            for j in range(n_arrays):
                agg[j] += w * arrays[j]

        return self._merge_layer_update(active, agg)
