# Why Go Full, When You Have Momentum?

Momentum-based similarity weighting for communication-efficient partial-update federated learning.

This repository implements and benchmarks a family of federated learning strategies built on **Flower** + **PyTorch**, centered on a new headline algorithm — **FedPartSAM-m** — which combines layer-wise partial training, server-side Adam, and cosine-similarity weighting on clients' Adam first-moment vectors.

> See `PROJECT_SUMMARY.md` for the full research write-up (motivation, related work, algorithm details).

## Headline contribution

**FedPartSAM-m** combines three ingredients:

1. **Partial layer-wise training** (FedPart, NeurIPS 2024) — only the active layer is trained per round
2. **Server-side Adam** on aggregated client deltas (FedOpt, ICLR 2021)
3. **Cosine-similarity weighting** on clients' Adam *first-moment* vectors (`m`), with a softmax temperature τ — the novel piece

The hypothesis: a client's Adam momentum encodes its optimization *trajectory* across rounds, so it is a richer similarity signal than a single-round parameter delta.

## Strategies

| ID | Class | Description |
|---|---|---|
| `fedavg` | `FedAvgStrategy` | Standard FedAvg (full model, weighted average) |
| `fedprox` | `FedAvgStrategy` + `proximal_mu>0` | FedAvg with proximal term — no separate class |
| `fedadam` | `FedAdamStrategy` | Full-model + server-side Adam (FedOpt baseline) |
| `fedpartavg` | `FedPartAvgStrategy` | Layer-wise partial training (Wang et al. 2024) |
| `fedpartadam` | `FedPartAdamStrategy` | Partial + server Adam, uniform weights (ablation) |
| `fedpartsam_m` | `FedPartSAMStrategy(similarity_source="momentum")` | **Headline**: partial + server Adam + Adam-`m` cosine similarity |
| `fedpartsam_d` | `FedPartSAMStrategy(similarity_source="delta")` | Ablation: same as above but similarity on parameter deltas |
| `fedpseudogradsim` | `FedPseudoGradSimStrategy` | Partial + delta similarity, no server Adam (ablation) |

## Repository layout

```
src/
├── config.py          # ExperimentConfig dataclass
├── device.py          # CUDA → MPS → CPU detection
├── data.py            # CIFAR-10/100 federated loaders (IID + Dirichlet)
├── comm.py            # Actual-byte communication accounting
├── serialize.py       # base64+pickle for transmitting momentum arrays
├── state.py           # Persistent client-side state across fit() calls
├── layers.py          # LayerSpec — logical layer groupings (CNN: 8, ResNet-18: 6)
├── params.py          # get/set parameters, freeze layers
├── train.py           # Local training (handles FedAvg + FedProx via flag)
├── eval_fn.py         # Centralized server-side evaluation
├── plotting.py        # Publication-quality plots
├── models/
│   ├── cnn.py             # Custom CNN (CIFAR-10 baseline)
│   └── resnet18_gn.py     # ResNet-18 with GroupNorm (FL-friendly)
├── strategies/        # 8 strategies — see table above
└── clients/
    ├── base.py            # FullModelClient + PartialClient
    └── fedpartsam.py      # FedPartSAMClient with persistent local Adam state

experiments/
└── run_all.py         # Resumable experiment runner
```

## Setup

```bash
conda create -n fedmom python=3.12
conda activate fedmom
pip install -r requirements.txt
```

## Running experiments

The runner is **resumable** — each completed config is saved immediately to `results/experiment_<n>.pkl`, and re-running the same command picks up where it left off.

```bash
python experiments/run_all.py                          # all 5 experiments
python experiments/run_all.py --experiment 1 3         # subset
python experiments/run_all.py --strategy fedpartsam_m  # one strategy across all
python experiments/run_all.py --seeds 0 1 2            # custom seeds
```

For long unattended runs on macOS:
```bash
nohup caffeinate -i python3 -u experiments/run_all.py > results/log.txt 2>&1 &
```

## Experiments

The primary setup is **ResNet-18 (GroupNorm) + CIFAR-100** with 50 rounds, 10 clients, 3 seeds.

| # | Name | Setup |
|---|---|---|
| 1 | IID sanity check | All 8 strategies, 100 % participation, IID |
| 2 | Non-IID sweep | All 8 strategies × Dirichlet α ∈ {0.1, 0.3, 1.0} |
| 3 | Realistic dropout | All 8 strategies, **50 % participation**, α = 0.3 |
| 4 | Temperature ablation | `fedpartsam_m` and `fedpartsam_d` × τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 1e7}, α = 0.3, 50 % participation |
| 5 | CIFAR-10 + CNN cross-validation | `fedavg`, `fedpartavg`, `fedpartsam_m`, `fedpartsam_d`, α = 0.3, 30 rounds |

Experiment 5 connects back to the original masters project (CNN + CIFAR-10).

## Running on Google Colab

```python
!git clone https://github.com/ViragSimon/layer_momentum.git
%cd layer_momentum
!pip install -q flwr==1.15.1 flwr-datasets==0.5.0 ray==2.31.0

# Mount Drive so results survive disconnects
from google.colab import drive
drive.mount('/content/drive')

!python experiments/run_all.py --experiment 1 \
    --output-dir /content/drive/MyDrive/layer_momentum/results
```

Switch the runtime to a T4 GPU (Runtime → Change runtime type) — `src/device.py` will auto-detect CUDA.

## Result format

Each `results/experiment_<n>.pkl` is a list of run dicts:

```python
{
    "config": {...},                # asdict(ExperimentConfig)
    "results": {
        "round_results": {round: {"total_bytes", "down_bytes", "up_bytes",
                                   "num_clients", "active_layer"?}},
        "eval_results":  {round: {"loss", "accuracy"}},
        "weight_history": [...],    # FedPartSAM / FedPseudoGradSim only
        "tau": float,               # FedPartSAM / FedPseudoGradSim only
    },
    "elapsed_seconds": float,
}
```

## Plotting

`src/plotting.py` provides:

- `plot_accuracy_vs_round` — convergence curves with std error bands
- `plot_accuracy_vs_bytes` — **the key efficiency plot** (accuracy vs cumulative MB)
- `plot_temperature_sweep` — τ ablation for FedPartSAM
- `plot_weight_distribution` — per-round client weight box plots
- `make_results_table` — markdown table of final accuracy + comm cost

## Key design decisions

1. **`build_strategy(config, ...)` and `build_client_fn(config, ...)`** in `src/strategies/__init__.py` and `src/clients/__init__.py` are the two factory entry points the runner uses. Adding a new strategy means registering it there.
2. **Communication cost is actual transmitted bytes** (`numpy_bytes` in `src/comm.py`), not Python object overhead.
3. **Centralized evaluation only** — `BaseStrategy.evaluate()` runs the global model on the test set after each round. Simpler and avoids per-client eval noise.
4. **FedProx is not a separate class** — set `proximal_mu > 0` on any strategy and clients automatically apply the proximal term during local training.
5. **Layer-wise schedule** lives in `PartialStrategy.active_layer_for_round()`: `warmup_rounds` full-model rounds, then cycle through layers with `rounds_per_layer` rounds each.
6. **FedPartSAM-m's optimizer state** — clients persist their own `exp_avg_sq` (`v`) locally across rounds; only `exp_avg` (`m`) is exchanged with the server. This avoids the `m / sqrt(0) = inf` bug and preserves per-client gradient-variance estimates.
7. **Server-side Adam uses `eps = 1e-3`** (Reddi et al. 2021 default), not the PyTorch default `1e-8` — the latter causes numerical explosion when applied to FL pseudo-gradients.
