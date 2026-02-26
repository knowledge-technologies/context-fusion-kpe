"""Training utilities for keyphrase extraction.

This module provides training and validation utilities:

- validate: Run validation on a dataset and compute loss + keyphrase metrics
"""

from keyword_pecker.trainer.validate import validate

__all__ = ["validate"]
