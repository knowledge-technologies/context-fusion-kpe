"""Early stopping utility for training.

This module provides a simple early stopping mechanism that monitors a metric
and signals when training should stop after no improvement for a given patience.
"""

from __future__ import annotations

from collections.abc import Mapping


class EarlyStopping:
    """Monitor a metric and signal when training should stop.

    Args:
        metric: Name of the metric to monitor (key in metrics dict).
        patience: Number of checks without improvement before stopping.
        delta: Minimum change to qualify as an improvement.
        mode: "max" for higher-is-better (e.g., F1), "min" for lower-is-better (e.g., loss).

    Attributes:
        best_score: Best metric value observed so far.
        best_step: Step number when best score was recorded.
        counter: Number of checks since last improvement.
    """

    def __init__(
        self,
        metric: str,
        patience: int,
        delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")

        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score: float | None = None
        self.best_step: int = 0
        self.counter: int = 0

    def __call__(self, metrics: Mapping[str, float], step: int = 0) -> bool:
        """Check if training should stop.

        Args:
            metrics: Mapping of metric values, must contain self.metric key.
            step: Current step number (for tracking best_step).

        Returns:
            True if training should stop, False otherwise.

        Raises:
            KeyError: If self.metric is not in metrics mapping.
        """
        current = metrics[self.metric]

        if self.best_score is None:
            self.best_score = current
            self.best_step = step
            return False

        improved = self._is_improvement(current)

        if improved:
            self.best_score = current
            self.best_step = step
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "max":
            return current >= self.best_score + self.delta
        else:
            return current <= self.best_score - self.delta

    def reset(self) -> None:
        """Reset state for a new training run."""
        self.best_score = None
        self.best_step = 0
        self.counter = 0
