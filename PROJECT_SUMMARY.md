# Why Go Full, When You Have Momentum?

## Momentum-Based Similarity Weighting for Communication-Efficient Partial-Update FL

---

## TL;DR

This project investigates whether **clients' Adam first-moment vectors** can serve as a useful signal for weighting client contributions in federated learning. The core thesis is that momentum encodes a client's optimization *trajectory* across many rounds, making it a richer similarity signal than single-round parameter deltas.

The headline algorithm is **FedPartSAM-m**:
1. **Partial layer-wise training** (FedPart, NeurIPS 2024) — only the active layer is trained per round, dramatically reducing communication
2. **Server-side Adam** on the aggregated client deltas (FedOpt, ICLR 2021) — provides acceleration
3. **Cosine-similarity weighting** — clients are weighted by the similarity of their Adam first-moment to the previous round's aggregated global momentum, with a softmax-temperature τ

The novelty is the third component: no prior work uses optimizer momentum vectors as a similarity signal for client weighting in aggregation.

---

## 1. Core Problem

Federated learning has two recurring problems:
1. **Communication cost** — sending the full model every round is expensive
2. **Heterogeneity** — clients with non-IID data drift in different directions, hurting convergence

Existing approaches address these separately:
- **FedPart** addresses (1) by training only one layer per round
- **FedAdam, FedAvgM** address (2) by adding server-side momentum
- **FedAWA, L-DAWA** address (2) by weighting clients by parameter similarity

This project asks: **what if we combine all three?** And specifically, what if we use *Adam momentum* (rather than parameter deltas) as the similarity signal?

---

## 2. Why Momentum?

A client's parameter delta `Δ = w_after - w_before` only captures what happened *this round* — it's noisy under non-IID data and resets when a client comes back from a dropout.

A client's Adam first-moment `m = β₁·m_prev + (1-β₁)·g` is an exponential moving average of all gradients seen so far. It encodes the client's optimization *trajectory* and:
- Smooths out per-round gradient noise
- Persists across dropout (a returning client knows where it was heading)
- For partial training, retains a memory of past update directions for frozen layers

**The central hypothesis: momentum-based similarity is a better signal than delta-based similarity for client weighting in partial-update FL.**

---

## 3. Algorithm

### FedPartSAM-m (headline)

```
for each round t:
  active_layer ← active_layer_schedule(t)

  for each sampled client i:
    receive (w_active, m_global_active) from server
    load m_global_active into local Adam's exp_avg
    train locally for E epochs (Adam, freezing all but active_layer)
    return (w_i, m_i)

  # Server side
  similarities = [cos(m_i, m_ref[active_layer]) for each i]
  weights = softmax(similarities / τ)
  Δ_agg = Σ_i weights[i] · (w_i - w_active)

  # Server-side Adam
  M ← β₁M + (1-β₁)Δ_agg
  V ← β₂V + (1-β₂)Δ_agg²
  w_new ← w_active + η · M / (√V + ε)

  m_ref[active_layer] ← mean(m_i)  # for next round's similarity
```

Two flavours, controlled by `similarity_source`:
- **`momentum`** — similarity computed on Adam's first moment (the headline)
- **`delta`** — similarity computed on parameter deltas (an ablation; same architecture, different signal)

---

## 4. Comparison Table (planned)

| Strategy | Trains | Server momentum | Similarity weighting | Comm cost |
|---|---|---|---|---|
| FedAvg | Full model | No | No | Full model |
| FedProx | Full model + prox term | No | No | Full model |
| FedAdam | Full model | Yes | No | Full model |
| FedPartAvg | One layer/round | No | No | One layer |
| FedPartAdam | One layer/round | Yes | No | One layer |
| FedPseudoGradSim | One layer/round | No | On deltas | One layer |
| **FedPartSAM-Δ** | One layer/round | Yes | On deltas | One layer |
| **FedPartSAM-m** | One layer/round | Yes | **On Adam m** | One layer + m |

The key paper comparison is **FedPartSAM-m vs FedPartSAM-Δ** — same architecture, different similarity signal. If -m wins, the momentum hypothesis is validated.

---

## 5. Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Model | Custom CNN (~200K params, 8 layer pairs) |
| Clients | 10 |
| Local epochs | 8 |
| Batch size | 32 |
| Rounds | 30 |
| Seeds | 42, 123, 456 |
| Compute | Single MacBook with MPS backend |

### 5 Experiments (`experiments/run_all.py`)

| # | Name | Setting | Strategies |
|---|------|---------|-----------|
| 1 | IID sanity check | α=∞, 100% participation | All 8 |
| 2 | Non-IID sweep | α∈{0.1, 0.3, 1.0}, 100% participation | All 8 |
| 3 | Realistic | α=0.3, 50% participation | All 8 |
| 4 | Temperature ablation | α=0.3, 50%, τ∈{0.1, 0.5, 1, 2, 5, ∞} | FedPartSAM-m, -Δ |
| 5 | Layer selection ablation | α=0.3, 50%, sequential vs adaptive | FedPartSAM-m |

---

## 6. Implementation Notes

The codebase was refactored to address several bugs and inefficiencies:

1. **Communication cost was measured wrong** — original used `sys.getsizeof` (Python object overhead). New `comm.py` counts actual numpy bytes. Real cost reduction from FedPart is now visible in plots (300×+ for partial rounds).

2. **FedPartSAM-m client had m/sqrt(0)=inf bug** — initial implementation patched only Adam's `exp_avg`, leaving `exp_avg_sq=0`, causing numerical explosion on first step. Fix: clients persist their own `exp_avg_sq` across rounds.

3. **Server Adam used PyTorch default `eps=1e-8`** — Reddi et al. 2021 use `1e-3`; the default causes blow-up when applied to FL pseudo-gradients.

4. **Layer-wise schedule was hardcoded** — refactored to use `warmup_rounds` and `rounds_per_layer` from config; cycles indefinitely.

5. **Eight legacy strategies dropped** — FedMom1, FedMom2, LocalAdam, FedPseudoGradient, etc. were exploratory implementations from the original masters project. Replaced with the cleaner 7-strategy core.

---

## 7. Tech Stack

- Python 3.12
- PyTorch 2.6 (MPS backend)
- Flower 1.15
- flwr-datasets 0.5
- CIFAR-10 via HuggingFace datasets
