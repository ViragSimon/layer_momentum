"""Device detection — CUDA → MPS → CPU.

Order matters: CUDA first so Colab/cloud GPUs are picked up; MPS as a fallback
for local Apple Silicon development; CPU as a last resort.
"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
