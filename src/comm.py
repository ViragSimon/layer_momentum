"""
Communication cost utilities.

The Flower default `Parameters` object uses `sys.getsizeof` which only counts
Python object overhead, not the actual tensor bytes. These helpers count the
true number of bytes that would be transmitted over the wire.
"""

from typing import Iterable, List, Union

import numpy as np
from flwr.common import Parameters, parameters_to_ndarrays


def numpy_bytes(arrays: Iterable[np.ndarray]) -> int:
    """Total wire bytes for a list of numpy arrays (no Python overhead)."""
    return sum(int(a.nbytes) for a in arrays)


def parameters_bytes(params: Parameters) -> int:
    """Total wire bytes for a Flower Parameters object."""
    return sum(len(t) for t in params.tensors)


def serialized_bytes(serialized: Union[str, bytes, None]) -> int:
    """Total wire bytes for a base64-encoded payload (e.g., momentum state)."""
    if serialized is None:
        return 0
    if isinstance(serialized, str):
        return len(serialized.encode("utf-8"))
    return len(serialized)
