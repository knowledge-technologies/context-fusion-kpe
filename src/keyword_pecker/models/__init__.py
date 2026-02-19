"""Model components for keyphrase extraction.

This module provides the neural network architectures for keyphrase extraction:

- BaseEncoder: Pretrained SBERT model wrapper for word embeddings
- RefinementHead: Custom transformer head for token-level classification
- KeyphraseModel: Combined end-to-end model

Typical usage:
    from keyword_pecker.models import KeyphraseModel

    model = KeyphraseModel()
    logits = model(input_ids, attention_mask, sentence_counts)
"""

from .base_encoder import BaseEncoder, mean_pooling
from .keyphrase_model import KeyphraseModel
from .refinement_head import RefinementHead

__all__ = [
    "BaseEncoder",
    "KeyphraseModel",
    "RefinementHead",
    "mean_pooling",
]
