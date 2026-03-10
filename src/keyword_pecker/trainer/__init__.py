"""Training utilities for keyphrase extraction.

This module provides training and evaluation utilities:

- evaluate_model: Run evaluation on a dataset split and compute loss + keyphrase metrics
"""

from keyword_pecker.trainer.validate import evaluate_model

__all__ = ["evaluate_model"]
