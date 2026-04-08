"""Federated learning clients."""

from typing import Callable

from flwr.client import Client, NumPyClient
from flwr.common import Context

from ..config import ExperimentConfig
from ..device import DEVICE
from .base import FullModelClient, PartialClient
from .fedpartsam import FedPartSAMClient


def build_client_fn(config: ExperimentConfig, model_fn: Callable, loader_fn: Callable) -> Callable[[Context], Client]:
    """Return a Flower `client_fn` for the given strategy.

    Parameters
    ----------
    config : ExperimentConfig
    model_fn : Callable[[], nn.Module]
        Returns a fresh model instance.
    loader_fn : Callable[[int, int], (DataLoader, DataLoader, DataLoader)]
        `(partition_id, num_partitions) -> (train, val, test)`.
    """
    name = config.strategy.lower()

    def client_fn(ctx: Context) -> Client:
        partition_id = ctx.node_config["partition-id"]
        num_partitions = ctx.node_config["num-partitions"]
        train_loader, val_loader, _ = loader_fn(partition_id, num_partitions)
        model = model_fn().to(DEVICE)

        if name in ("fedavg", "fedprox", "fedadam"):
            return FullModelClient(
                partition_id=partition_id,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.local_epochs,
                local_lr=config.local_lr,
            ).to_client()

        if name == "fedpartsam_m":
            return FedPartSAMClient(
                partition_id=partition_id,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.local_epochs,
                local_lr=config.local_lr,
                context=ctx,
                model_name=config.model,
            ).to_client()

        # All other partial strategies use the standard PartialClient
        if name in ("fedpartavg", "fedpartadam", "fedpartsam_d", "fedpseudogradsim"):
            return PartialClient(
                partition_id=partition_id,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.local_epochs,
                local_lr=config.local_lr,
                context=ctx,
                model_name=config.model,
            ).to_client()

        raise ValueError(f"No client for strategy: {name}")

    return client_fn


__all__ = ["FullModelClient", "PartialClient", "FedPartSAMClient", "build_client_fn"]
