"""
Client-side persistent state management.

Wraps Flower's `context.state.parameters_records` with a clean interface
for storing and retrieving lists of numpy arrays across rounds. Used by
clients that need to persist model parameters or optimizer state between
fit() calls (e.g., layer-wise training, FedPartSAM-m).
"""

from typing import List, Optional

import numpy as np
from flwr.common import Context, ParametersRecord, array_from_numpy


class ClientStateStore:
    """Persistent key→arrays store backed by Flower's context state.

    Each `key` maps to an ordered list of numpy arrays (typically the model's
    full state_dict tensors, or an optimizer's momentum vectors).

    Example
    -------
    >>> store = ClientStateStore(context)
    >>> store.save("model", [w1, b1, w2, b2])
    >>> arrays = store.load("model")
    """

    def __init__(self, context: Context):
        self._records = context.state.parameters_records

    def has(self, key: str) -> bool:
        return key in self._records

    def save(self, key: str, arrays: List[np.ndarray]) -> None:
        record = ParametersRecord()
        for i, arr in enumerate(arrays):
            record[f"_{i}"] = array_from_numpy(np.asarray(arr))
        self._records[key] = record

    def load(self, key: str) -> Optional[List[np.ndarray]]:
        if key not in self._records:
            return None
        record = self._records[key]
        return [record[f"_{i}"].numpy() for i in range(len(record))]
