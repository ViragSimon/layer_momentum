"""
Publication-quality plotting for FL experiments.

Result format expected (from `strategy.get_results()`):
    {
      "round_results": {round: {"down_bytes", "up_bytes", "total_bytes",
                                 "num_clients", "active_layer"?}},
      "eval_results": {round: {"loss", "accuracy"}},
      "weight_history": [...],   # FedPartSAM/FedPseudoGradSim only
      "tau": float,              # FedPartSAM/FedPseudoGradSim only
      "similarity_source": str,  # FedPartSAM only
    }
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Wong colorblind-friendly palette
COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedadam": "FedAdam",
    "fedpartavg": "FedPartAvg",
    "fedpartadam": "FedPartAdam",
    "fedpartsam_d": "FedPartSAM-Δ",
    "fedpartsam_m": "FedPartSAM-m",
    "fedpseudogradsim": "FedPseudoGradSim",
}


def load_experiment(path: str) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Data extraction helpers ──────────────────────────────────────────────

def _eval_curve(run: dict) -> tuple[np.ndarray, np.ndarray]:
    eval_results = run["results"]["eval_results"]
    rounds = sorted(eval_results.keys())
    accs = np.array([eval_results[r]["accuracy"] for r in rounds])
    return np.array(rounds), accs


def _bytes_curve(run: dict) -> np.ndarray:
    """Cumulative bytes after each round (matching the eval rounds)."""
    round_results = run["results"]["round_results"]
    rounds = sorted(round_results.keys())
    bytes_per_round = [round_results[r].get("total_bytes", 0) for r in rounds]
    return np.cumsum(bytes_per_round)


def _group_by_strategy(runs: List[dict], filters: Optional[dict] = None) -> Dict[str, List[dict]]:
    """Group runs by strategy, optionally filtering by config fields."""
    groups = defaultdict(list)
    for r in runs:
        cfg = r["config"]
        if filters:
            if not all(cfg.get(k) == v for k, v in filters.items()):
                continue
        groups[cfg["strategy"]].append(r)
    return dict(groups)


def _stack_curves(runs: List[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rounds, mean_acc, std_acc) across seeds."""
    all_accs = []
    rounds = None
    for r in runs:
        rd, accs = _eval_curve(r)
        if rounds is None:
            rounds = rd
        # pad/truncate to match length (in case some runs have different lengths)
        n = min(len(rounds), len(accs))
        all_accs.append(accs[:n])
        rounds = rounds[:n]
    arr = np.array(all_accs)
    return rounds, arr.mean(axis=0), arr.std(axis=0)


# ── Plot functions ────────────────────────────────────────────────────────

def plot_accuracy_vs_round(
    runs: List[dict], title: str = "Test Accuracy vs Round",
    save_path: Optional[str] = None, filters: Optional[dict] = None,
):
    groups = _group_by_strategy(runs, filters)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (strat, strat_runs) in enumerate(sorted(groups.items())):
        rd, mean, std = _stack_curves(strat_runs)
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.plot(rd, mean, color=c, marker=m, markersize=4, linewidth=1.5,
                label=label, markevery=max(1, len(rd) // 10))
        ax.fill_between(rd, mean - std, mean + std, alpha=0.15, color=c)

    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_accuracy_vs_bytes(
    runs: List[dict], title: str = "Test Accuracy vs Communication Cost",
    save_path: Optional[str] = None, filters: Optional[dict] = None,
):
    """The key efficiency plot: accuracy as a function of cumulative bytes."""
    groups = _group_by_strategy(runs, filters)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (strat, strat_runs) in enumerate(sorted(groups.items())):
        # Average accuracy across seeds at each round
        rd, mean_acc, std_acc = _stack_curves(strat_runs)
        # Average bytes-curve across seeds
        all_bytes = np.array([_bytes_curve(r)[:len(rd)] for r in strat_runs])
        bytes_mean = all_bytes.mean(axis=0) / 1e6  # MB

        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.plot(bytes_mean, mean_acc, color=c, marker=m, markersize=4,
                linewidth=1.5, label=label, markevery=max(1, len(bytes_mean) // 10))
        ax.fill_between(bytes_mean, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=c)

    ax.set_xlabel("Cumulative Communication (MB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_temperature_sweep(
    runs: List[dict], strategy: str = "fedpartsam_m",
    title: Optional[str] = None, save_path: Optional[str] = None,
):
    """Final accuracy vs temperature τ for a single strategy (FedPartSAM family)."""
    tau_groups = defaultdict(list)
    for r in runs:
        if r["config"]["strategy"] != strategy:
            continue
        tau_groups[r["config"]["tau"]].append(r)

    taus = sorted(tau_groups.keys())
    means, stds = [], []
    for tau in taus:
        finals = []
        for r in tau_groups[tau]:
            ev = r["results"]["eval_results"]
            last = max(ev.keys())
            finals.append(ev[last]["accuracy"])
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(taus))
    labels = [f"{t:.1f}" if t < 100 else "uniform" for t in taus]
    ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS[0], capsize=5, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Temperature τ")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(title or f"{STRATEGY_LABELS.get(strategy, strategy)}: Temperature ablation")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_weight_distribution(run: dict, title: Optional[str] = None, save_path: Optional[str] = None):
    """Box plot of per-round client weights from a FedPartSAM/FedPseudoGradSim run."""
    weight_history = run["results"].get("weight_history", [])
    if not weight_history:
        print("No weight history in this run")
        return None

    rounds = [w["round"] for w in weight_history]
    weights = [w["weights"] for w in weight_history]
    n_clients = len(weights[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(weights, positions=rounds, widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.5)

    ax.axhline(y=1.0 / n_clients, color="gray", linestyle="--",
               linewidth=1, label=f"Uniform (1/{n_clients})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Client Weight")
    ax.set_title(title or "Client weight distribution per round")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def make_results_table(runs: List[dict], filters: Optional[dict] = None) -> str:
    """Return a markdown table summarizing final accuracy + total comm cost."""
    groups = _group_by_strategy(runs, filters)
    rows = []
    for strat in sorted(groups.keys()):
        finals = []
        bytes_totals = []
        for r in groups[strat]:
            ev = r["results"]["eval_results"]
            last = max(ev.keys())
            finals.append(ev[last]["accuracy"])
            rr = r["results"]["round_results"]
            bytes_totals.append(sum(rr[k].get("total_bytes", 0) for k in rr))
        acc_mean, acc_std = np.mean(finals), np.std(finals)
        bytes_mean = np.mean(bytes_totals) / 1e6
        label = STRATEGY_LABELS.get(strat, strat)
        rows.append(f"| {label} | {acc_mean:.4f} ± {acc_std:.4f} | {bytes_mean:.1f} MB |")

    header = "| Strategy | Final Accuracy | Total Comm. |\n|---|---|---|\n"
    return header + "\n".join(rows)
