"""FedAvg strategy. Also supports FedProx via `proximal_mu > 0`."""

from .base import BaseStrategy


class FedAvgStrategy(BaseStrategy):
    """Standard FedAvg: weighted average of client parameters by num_examples.

    Pass `proximal_mu > 0` to enable FedProx behaviour (server logic is identical;
    the proximal term is applied client-side via the `proximal_mu` config field).
    """
    pass  # Default `_aggregate_round` in BaseStrategy is exactly FedAvg.
