"""
Lightweight serialization for transmitting numpy arrays in Flower's `metrics` /
`config` dicts (which only support scalars and bytes/strings).

We use base64-encoded pickled lists of float16 arrays to keep the wire footprint
small while preserving array structure. Used to ship Adam first-moments for
FedPartSAM-m and similar.
"""

import base64
import pickle
from typing import List, Optional

import numpy as np


def encode_arrays(arrays: List[np.ndarray]) -> str:
    """Encode a list of numpy arrays as a base64 string (float16 internally)."""
    payload = [a.astype(np.float16) for a in arrays]
    return base64.b64encode(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def decode_arrays(encoded: Optional[str]) -> List[np.ndarray]:
    """Decode a base64 string back into float32 numpy arrays."""
    if not encoded:
        return []
    payload = pickle.loads(base64.b64decode(encoded))
    return [a.astype(np.float32) for a in payload]
