"""Base encoder module wrapping a pretrained Sentence-BERT model."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Apply mean pooling over token embeddings using attention mask.

    Args:
        token_embeddings: Token-level embeddings of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Mask of shape (batch_size, seq_len) indicating valid tokens.

    Returns:
        Pooled embeddings of shape (batch_size, hidden_dim).
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class BaseEncoder(nn.Module):
    """Wrapper around a pretrained Sentence-BERT model.

    Provides word-level embeddings and sentence-level embeddings via mean pooling.
    Supports freezing/unfreezing for differential learning rate training.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        freeze: bool = False,
    ):
        """Initialize the base encoder.

        Args:
            model_name: HuggingFace model identifier for the pretrained encoder.
            freeze: If True, freeze all parameters (no gradient updates).
        """
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_model = self.encoder.config.hidden_size

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode input tokens to word-level embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Word-level embeddings of shape (batch_size, seq_len, d_model).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
