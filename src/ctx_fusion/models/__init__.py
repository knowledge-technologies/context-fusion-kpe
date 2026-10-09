"""Model components for keyphrase extraction.

This module provides the neural network architectures for keyphrase extraction:

- BaseEncoder: Pretrained SBERT model wrapper for word embeddings
- RefinementHead: Custom transformer head for token-level classification
- KeyphraseModel: Combined end-to-end model

Typical usage:
    from ctx_fusion.models import KeyphraseModel

    model = KeyphraseModel()
    logits = model(input_ids, attention_mask, sentence_counts)
"""

from .base_encoder import BaseEncoder, mean_pooling
from .inference import predict_keyphrases
from .keyphrase_model import KeyphraseModel
from .refinement_encoder import RefinementEncoder

__all__ = [
    "BaseEncoder",
    "KeyphraseModel",
    "RefinementEncoder",
    "mean_pooling",
    "predict_keyphrases",
]
