"""Training utilities for keyphrase extraction.

This module provides training and evaluation utilities:

- evaluate_model: Run evaluation on a dataset split and compute loss + keyphrase metrics
- EarlyStopping: Monitor a metric and signal when training should stop
- train: Full training routine with two-level batching and early stopping
"""

from ctx_fusion.trainer.early_stopping import EarlyStopping
from ctx_fusion.trainer.train import train
from ctx_fusion.trainer.validate import evaluate_model

__all__ = ["EarlyStopping", "evaluate_model", "train"]
