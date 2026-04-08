"""Experiment configuration dataclass."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single FL experiment run.

    Used by the experiment runner to specify a strategy + dataset + hyperparameters.
    """

    # ─── Strategy ─────────────────────────────────────────────────────────
    strategy: str = "fedavg"
    # Supported strategies:
    #   "fedavg"          — full-model FedAvg
    #   "fedprox"         — full-model FedAvg + proximal term
    #   "fedadam"         — full-model + server Adam
    #   "fedpartavg"      — partial layer-wise + uniform aggregation
    #   "fedpartadam"     — partial + server Adam (uniform weights)
    #   "fedpartsam_m"    — partial + server Adam + momentum-similarity weighting (HEADLINE)
    #   "fedpartsam_d"    — partial + server Adam + delta-similarity weighting (ablation)
    #   "fedpseudogradsim" — partial + delta-similarity (no server Adam, ablation)

    # ─── Dataset & model ──────────────────────────────────────────────────
    dataset: str = "cifar10"      # cifar10, cifar100
    model: str = "cnn"            # cnn, resnet18

    # ─── FL setup ─────────────────────────────────────────────────────────
    num_clients: int = 10
    participation_rate: float = 1.0   # fraction of clients sampled per round
    num_rounds: int = 50
    local_epochs: int = 8
    batch_size: int = 32
    local_lr: float = 1e-3            # client Adam learning rate

    # ─── Data heterogeneity ───────────────────────────────────────────────
    dirichlet_alpha: Optional[float] = None  # None=IID, float=non-IID

    # ─── Layer-wise training ──────────────────────────────────────────────
    warmup_rounds: int = 2            # full-model rounds before layer-wise begins
    rounds_per_layer: int = 2         # consecutive rounds per layer
    layer_selection: str = "sequential"  # sequential | adaptive

    # ─── Server-side optimizer (FedAdam, FedPartSAM) ──────────────────────
    server_lr: float = 0.01
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_eps: float = 1e-3          # Reddi 2021 default; NOT 1e-8

    # ─── Similarity weighting (FedPartSAM, FedPseudoGradSim) ──────────────
    tau: float = 1.0                  # softmax temperature

    # ─── FedProx ──────────────────────────────────────────────────────────
    proximal_mu: float = 0.1

    # ─── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    @property
    def clients_per_round(self) -> int:
        return max(1, int(round(self.num_clients * self.participation_rate)))

    @property
    def is_iid(self) -> bool:
        return self.dirichlet_alpha is None

    def to_dict(self) -> dict:
        return asdict(self)
