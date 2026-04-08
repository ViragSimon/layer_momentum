"""
Federated CIFAR-10 / CIFAR-100 data loaders.

Returns per-client (train, val, test) DataLoaders. Supports IID partitioning
(uniform random) and non-IID partitioning (Dirichlet with configurable alpha).
"""

from functools import partial
from typing import Callable, Tuple

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms


# Per-dataset normalization stats
_NORMALIZE = {
    "cifar10": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _make_transform(dataset: str, label_col: str = "label"):
    mean, std = _NORMALIZE[dataset]
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def apply(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        # Normalize label column name so downstream training code is dataset-agnostic
        if label_col != "label":
            batch["label"] = batch[label_col]
        return batch

    return apply


def _build_loaders(
    dataset: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders for a single client partition.

    Both CIFAR-10 and CIFAR-100 are normalized so that batches have an "img"
    and "label" key (CIFAR-100's native key is "fine_label").
    """
    label_col = "fine_label" if dataset == "cifar100" else "label"

    if alpha is None:
        partitioners = {"train": num_partitions}
    else:
        partitioners = {"train": DirichletPartitioner(
            num_partitions=num_partitions, alpha=alpha, partition_by=label_col,
        )}

    fds = FederatedDataset(dataset=dataset, partitioners=partitioners)
    partition = fds.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=42)

    apply = _make_transform(dataset, label_col)
    split = split.with_transform(apply)
    test_set = fds.load_split("test").with_transform(apply)

    train_loader = DataLoader(split["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(split["test"], batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def make_loader_fn(
    dataset: str = "cifar10",
    batch_size: int = 32,
    alpha: float = None,
) -> Callable[[int, int], Tuple[DataLoader, DataLoader, DataLoader]]:
    """Return a `(partition_id, num_partitions) -> loaders` function bound to a config.

    The returned function matches the signature Flower clients expect.
    """
    return partial(_build_loaders, dataset, batch_size=batch_size, alpha=alpha)
