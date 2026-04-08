"""
Experiment runner for all FL strategies.

Resumable: each completed run is saved immediately. Re-running with the same
command picks up where the previous invocation left off.

Usage
-----
    python experiments/run_all.py                          # all 5 experiments
    python experiments/run_all.py --experiment 1 3         # subset
    python experiments/run_all.py --strategy fedpartsam_m  # one strategy across all experiments
    python experiments/run_all.py --seeds 0 1 2            # custom seeds
"""

import argparse
import gc
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict
from typing import List

import torch

# Make `src` importable from the experiments folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

from src.clients import build_client_fn
from src.config import ExperimentConfig
from src.data import make_loader_fn
from src.device import DEVICE
from src.eval_fn import make_evaluate_fn
from src.models import build_model
from src.strategies import build_strategy

logging.getLogger("flwr").setLevel(logging.ERROR)


# ── Single-run execution ──────────────────────────────────────────────────

def run_single(config: ExperimentConfig) -> dict:
    print(f"\n{'='*60}")
    print(f" {config.strategy}  seed={config.seed}  alpha={config.dirichlet_alpha}  tau={config.tau}")
    print(f" clients={config.num_clients} part={config.participation_rate} rounds={config.num_rounds}")
    print(f"{'='*60}")

    # Build model factory and dataset loader
    def model_fn():
        return build_model(config.model, num_classes=100 if config.dataset == "cifar100" else 10)

    loader_fn = make_loader_fn(
        dataset=config.dataset,
        batch_size=config.batch_size,
        alpha=config.dirichlet_alpha,
    )

    # Build evaluation function from a global test loader
    _, _, test_loader = loader_fn(0, config.num_clients)
    eval_model = model_fn().to(DEVICE)
    evaluate_fn = make_evaluate_fn(test_loader, eval_model)

    # Build strategy and client_fn
    strategy = build_strategy(config, evaluate_fn=evaluate_fn, model_fn=model_fn)
    client_fn = build_client_fn(config, model_fn=model_fn, loader_fn=loader_fn)

    # Wrap into Flower apps
    def server_fn(_: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=config.num_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(client_fn=client_fn)

    # On a single CUDA GPU, reserve the whole device per client so Ray
    # serializes clients onto it instead of letting them collide. On MPS / CPU
    # Ray's GPU scheduler is irrelevant, so 0.0 is correct (and lets Ray run
    # clients in parallel up to num_cpus).
    num_gpus = 1.0 if DEVICE.type == "cuda" else 0.0

    start = time.time()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": num_gpus}},
    )
    elapsed = time.time() - start

    return {
        "config": asdict(config),
        "results": strategy.get_results(),
        "elapsed_seconds": elapsed,
    }


# ── Resume support ────────────────────────────────────────────────────────

def _config_key(cfg: dict) -> tuple:
    """Unique identifier for a config (strategy + seed + alpha + tau + layer_selection)."""
    return (
        cfg["strategy"], cfg["seed"], cfg.get("dirichlet_alpha"),
        cfg.get("tau"), cfg.get("layer_selection"),
    )


def _load_existing(path: str) -> List[dict]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def _run_experiment(exp_num: int, configs: List[ExperimentConfig], output_dir: str):
    out_path = os.path.join(output_dir, f"experiment_{exp_num}.pkl")
    results = _load_existing(out_path)
    done_keys = {_config_key(r["config"]) for r in results}

    print(f"\n{'#'*60}")
    print(f"# Experiment {exp_num}: {len(configs)} configs ({len(done_keys)} already done)")
    print(f"{'#'*60}")

    for cfg in configs:
        if _config_key(asdict(cfg)) in done_keys:
            continue
        try:
            result = run_single(cfg)
            results.append(result)
            with open(out_path, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f" ✓ saved ({len(results)} runs in {os.path.basename(out_path)})")
        except Exception as e:
            print(f" ✗ FAILED {cfg.strategy} seed={cfg.seed}: {e}")
            continue
        finally:
            # Aggressively reclaim memory between runs so Ray workers have
            # headroom on low-RAM nodes (e.g. Colab free tier). Without this,
            # driver + leftover actor state accumulates across strategies
            # until the OOMKiller starts slaughtering Flower workers mid-fit.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── Experiment definitions ────────────────────────────────────────────────

CORE_STRATEGIES = [
    "fedavg", "fedprox", "fedadam",
    "fedpartavg", "fedpartadam",
    "fedpartsam_d", "fedpartsam_m",
    "fedpseudogradsim",
]

# Default model + dataset for experiments 1-4. CIFAR-100 + ResNet-18 is the
# primary setup per the improvement plan; CIFAR-10 + custom CNN is experiment 5.
PRIMARY = dict(model="resnet18", dataset="cifar100", num_rounds=50)


def exp_1_iid(seeds: List[int]) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            strategy=s, num_clients=10, participation_rate=1.0,
            dirichlet_alpha=None, seed=seed, **PRIMARY,
        )
        for s in CORE_STRATEGIES for seed in seeds
    ]


def exp_2_noniid_sweep(seeds: List[int]) -> List[ExperimentConfig]:
    configs = []
    for alpha in [0.1, 0.3, 1.0]:
        for s in CORE_STRATEGIES:
            for seed in seeds:
                configs.append(ExperimentConfig(
                    strategy=s, num_clients=10, participation_rate=1.0,
                    dirichlet_alpha=alpha, seed=seed, **PRIMARY,
                ))
    return configs


def exp_3_dropout(seeds: List[int]) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            strategy=s, num_clients=10, participation_rate=0.5,
            dirichlet_alpha=0.3, seed=seed, **PRIMARY,
        )
        for s in CORE_STRATEGIES for seed in seeds
    ]


def exp_4_temperature(seeds: List[int]) -> List[ExperimentConfig]:
    configs = []
    for strat in ["fedpartsam_m", "fedpartsam_d"]:
        for tau in [0.1, 0.5, 1.0, 2.0, 5.0, 1e7]:
            for seed in seeds:
                configs.append(ExperimentConfig(
                    strategy=strat, num_clients=10, participation_rate=0.5,
                    dirichlet_alpha=0.3, tau=tau, seed=seed, **PRIMARY,
                ))
    return configs


def exp_5_cifar10_cnn(seeds: List[int]) -> List[ExperimentConfig]:
    """CIFAR-10 + custom CNN cross-validation (connects to original masters project)."""
    strategies = ["fedavg", "fedpartavg", "fedpartsam_m", "fedpartsam_d"]
    return [
        ExperimentConfig(
            strategy=s, model="cnn", dataset="cifar10",
            num_clients=10, participation_rate=0.5,
            num_rounds=30, dirichlet_alpha=0.3, seed=seed,
        )
        for s in strategies for seed in seeds
    ]


EXPERIMENTS = {
    1: ("IID sanity check (CIFAR-100 + ResNet-18)", exp_1_iid),
    2: ("Non-IID sweep α ∈ {0.1, 0.3, 1.0}", exp_2_noniid_sweep),
    3: ("Non-IID + 50% dropout (realistic)", exp_3_dropout),
    4: ("Temperature ablation (FedPartSAM-m vs -d)", exp_4_temperature),
    5: ("CIFAR-10 + custom CNN cross-validation", exp_5_cifar10_cnn),
}


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run federated learning experiments")
    parser.add_argument("--experiment", type=int, nargs="*", default=None,
                        help="Experiment numbers to run (default: 1 2 3 4 5)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Filter to runs of this strategy only")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override num_rounds on every config (for smoke tests)")
    parser.add_argument("--local-epochs", type=int, default=None,
                        help="Override local_epochs on every config (for smoke tests)")
    parser.add_argument("--warmup-rounds", type=int, default=None,
                        help="Override warmup_rounds on every config (for smoke tests)")
    parser.add_argument("--rounds-per-layer", type=int, default=None,
                        help="Override rounds_per_layer on every config (for smoke tests)")
    parser.add_argument("--num-clients", type=int, default=None,
                        help="Override num_clients on every config (use smaller values "
                             "on low-RAM nodes like Colab free tier)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nums = args.experiment or list(EXPERIMENTS.keys())

    for n in nums:
        if n not in EXPERIMENTS:
            print(f"Unknown experiment: {n}")
            continue
        name, fn = EXPERIMENTS[n]
        print(f"\n{'─'*60}\nExperiment {n}: {name}\n{'─'*60}")
        configs = fn(args.seeds)
        if args.strategy:
            configs = [c for c in configs if c.strategy == args.strategy]
        if args.rounds is not None:
            for c in configs:
                c.num_rounds = args.rounds
        if args.local_epochs is not None:
            for c in configs:
                c.local_epochs = args.local_epochs
        if args.warmup_rounds is not None:
            for c in configs:
                c.warmup_rounds = args.warmup_rounds
        if args.rounds_per_layer is not None:
            for c in configs:
                c.rounds_per_layer = args.rounds_per_layer
        if args.num_clients is not None:
            for c in configs:
                c.num_clients = args.num_clients
        _run_experiment(n, configs, args.output_dir)


if __name__ == "__main__":
    main()
