"""Combined keyphrase extraction model.

This module provides the full keyphrase extraction model that combines
the base encoder (SBERT) with the refinement head for end-to-end inference.
"""

import torch
import torch.nn as nn

from .base_encoder import BaseEncoder
from .refinement_encoder import RefinementEncoder


class KeyphraseModel(nn.Module):
    """End-to-end keyphrase extraction model.

    Combines a pretrained base encoder (SBERT) with a custom refinement head
    to produce token-level keyphrase predictions. Supports differential
    learning rates for fine-tuning the base encoder vs training the head.

    Architecture:
        Input tokens -> BaseEncoder -> word embeddings -> RefinementEncoder -> token logits
    """

    def __init__(
        self,
        base_encoder_name: str = BaseEncoder.DEFAULT_MODEL_NAME,
        freeze_base_encoder: bool = False,
        d_model: int = 768,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_position_embeddings: int = 768,
    ):
        """Initialize the keyphrase extraction model.

        Args:
            model_name: HuggingFace model identifier for the base encoder.
            freeze_encoder: If True, freeze base encoder parameters.
            d_model: Dimension of the model (should match base encoder).
            nhead: Number of attention heads in refinement head.
            num_layers: Number of transformer layers in refinement head.
            dim_feedforward: Feedforward dimension. Defaults to 4 * d_model.
            dropout: Dropout probability in refinement head.
            max_position_embeddings: Maximum sequence length for positional embeddings.
        """
        super().__init__()

        # Base encoder (pretrained SBERT)
        self.base_encoder = BaseEncoder(
            model_name=base_encoder_name,
            freeze=freeze_base_encoder,
        )

        # Verify d_model matches base encoder
        if d_model != self.base_encoder.d_model:
            raise ValueError(
                f"d_model ({d_model}) must match base encoder hidden size "
                f"({self.base_encoder.d_model})"
            )

        # Refinement head
        self.refinement_head = RefinementEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
        )

    @property
    def tokenizer(self):
        """Access the base encoder's tokenizer."""
        return self.base_encoder.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_counts: list[int],
    ) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs of shape (num_sentences, max_seq_len).
            attention_mask: Attention mask of shape (num_sentences, max_seq_len).
            sentence_counts: List of sentence counts per document.

        Returns:
            Token-level logits of shape (num_sentences, max_seq_len).
        """
        # Get word embeddings from base encoder
        word_embeddings = self.base_encoder(input_ids, attention_mask)

        # Refine with document context
        logits = self.refinement_head(word_embeddings, attention_mask, sentence_counts)

        return logits

    def get_param_groups(
        self,
        base_lr: float = 2e-5,
        head_lr: float = 5e-5,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """Create parameter groups for differential learning rates.

        Separates parameters into groups for:
        - Base encoder (lower learning rate for fine-tuning)
        - Refinement head (higher learning rate for training from scratch)

        Also separates bias/LayerNorm parameters (no weight decay) from others.

        Args:
            base_lr: Learning rate for base encoder parameters.
            head_lr: Learning rate for refinement head parameters.
            weight_decay: Weight decay for non-bias/LayerNorm parameters.

        Returns:
            List of parameter group dicts for optimizer initialization.
        """
        no_decay = {"bias", "LayerNorm.weight"}

        param_groups = [
            # Base encoder - with weight decay
            {
                "params": [
                    p
                    for n, p in self.base_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
                "lr": base_lr,
            },
            # Base encoder - no weight decay
            {
                "params": [
                    p
                    for n, p in self.base_encoder.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
            # Refinement head - with weight decay
            {
                "params": [
                    p
                    for n, p in self.refinement_head.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": head_lr,
            },
            # Refinement head - no weight decay
            {
                "params": [
                    p
                    for n, p in self.refinement_head.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": head_lr,
            },
        ]

        # Filter out empty groups
        return [g for g in param_groups if g["params"]]
