"""
Local training and evaluation routines.

Single `train_local` function handles all FL strategies (FedAvg, FedProx)
via optional kwargs. The optimizer is created externally and can be passed in
so that callers (e.g., FedPartSAM client) can manipulate Adam state directly.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .device import DEVICE


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 1e-3,
    proximal_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    verbose: bool = False,
) -> torch.optim.Optimizer:
    """Train a model locally for `num_epochs`.

    Parameters
    ----------
    model : nn.Module
        The model to train (in-place updates).
    train_loader : DataLoader
        Per-client training data.
    num_epochs : int
        Number of local epochs (E in FedAvg notation).
    optimizer : torch.optim.Optimizer, optional
        Pre-built optimizer. If None, a fresh Adam(lr=lr) is created. Pass an
        existing optimizer to preserve / control its state across rounds.
    lr : float
        Learning rate (only used if `optimizer` is None).
    proximal_mu : float
        FedProx coefficient. If > 0, adds (mu/2)*||w - w_global||^2 to the loss.
    global_params : List[torch.Tensor], optional
        Frozen global parameters for the proximal term. Required if proximal_mu > 0.
    verbose : bool
        If True, print per-epoch loss/accuracy.

    Returns
    -------
    torch.optim.Optimizer
        The (possibly created) optimizer with its updated state.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

    criterion = nn.CrossEntropyLoss()
    use_prox = proximal_mu > 0.0 and global_params is not None

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if use_prox:
                prox_term = sum(
                    (lp - gp).pow(2).sum()
                    for lp, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if verbose and total > 0:
            avg_loss = total_loss / max(1, len(train_loader))
            acc = correct / total
            print(f"  Local epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f} acc={acc:.4f}")

    return optimizer


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["img"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)
