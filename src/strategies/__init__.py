"""Federated learning strategies."""

from .base import BaseStrategy, PartialStrategy
from .fedavg import FedAvgStrategy
from .fedadam import FedAdamStrategy
from .fedpart import FedPartAvgStrategy
from .fedpartadam import FedPartAdamStrategy
from .fedpartsam import FedPartSAMStrategy
from .pseudograd import FedPseudoGradSimStrategy


def build_strategy(config, evaluate_fn, model_fn) -> BaseStrategy:
    """Instantiate a strategy from an ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig
    evaluate_fn : callable
        Server-side evaluation function (from `make_evaluate_fn`).
    model_fn : callable
        Zero-arg factory that returns a fresh model instance (used for initial params).
    """
    name = config.strategy.lower()
    common = dict(
        evaluate_fn=evaluate_fn,
        model_fn=model_fn,
        fraction_fit=config.participation_rate,
        fraction_evaluate=0.0,  # we use centralized evaluation only
        min_fit_clients=config.clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=config.num_clients,
        warmup_rounds=config.warmup_rounds,
        rounds_per_layer=config.rounds_per_layer,
        model_name=config.model,
    )

    if name == "fedavg":
        return FedAvgStrategy(**common)
    if name == "fedprox":
        return FedAvgStrategy(proximal_mu=config.proximal_mu, **common)
    if name == "fedadam":
        return FedAdamStrategy(
            server_lr=config.server_lr,
            server_beta1=config.server_beta1,
            server_beta2=config.server_beta2,
            server_eps=config.server_eps,
            **common,
        )
    if name == "fedpartavg":
        return FedPartAvgStrategy(**common)
    if name == "fedpartadam":
        return FedPartAdamStrategy(
            server_lr=config.server_lr,
            server_beta1=config.server_beta1,
            server_beta2=config.server_beta2,
            server_eps=config.server_eps,
            **common,
        )
    if name == "fedpartsam_m":
        return FedPartSAMStrategy(
            similarity_source="momentum",
            tau=config.tau,
            server_lr=config.server_lr,
            server_beta1=config.server_beta1,
            server_beta2=config.server_beta2,
            server_eps=config.server_eps,
            layer_selection=config.layer_selection,
            **common,
        )
    if name == "fedpartsam_d":
        return FedPartSAMStrategy(
            similarity_source="delta",
            tau=config.tau,
            server_lr=config.server_lr,
            server_beta1=config.server_beta1,
            server_beta2=config.server_beta2,
            server_eps=config.server_eps,
            layer_selection=config.layer_selection,
            **common,
        )
    if name == "fedpseudogradsim":
        return FedPseudoGradSimStrategy(tau=config.tau, **common)
    raise ValueError(f"Unknown strategy: {name}")


__all__ = [
    "BaseStrategy", "PartialStrategy",
    "FedAvgStrategy", "FedAdamStrategy",
    "FedPartAvgStrategy", "FedPartAdamStrategy", "FedPartSAMStrategy",
    "FedPseudoGradSimStrategy",
    "build_strategy",
]
