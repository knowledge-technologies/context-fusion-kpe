"""Training utilities for keyphrase extraction.

This module provides training and evaluation utilities:

- evaluate_model: Run evaluation on a dataset split and compute loss + keyphrase metrics
- EarlyStopping: Monitor a metric and signal when training should stop
"""

from keyword_pecker.trainer.early_stopping import EarlyStopping
from keyword_pecker.trainer.validate import evaluate_model

__all__ = ["EarlyStopping", "evaluate_model"]
