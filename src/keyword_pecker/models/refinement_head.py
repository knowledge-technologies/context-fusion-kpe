"""Refinement head module for keyphrase extraction.

This module implements the custom transformer head that processes word embeddings
with document context to produce token-level keyphrase predictions.
"""

import torch
import torch.nn as nn

from .base_encoder import mean_pooling


class RefinementHead(nn.Module):
    """Transformer-based refinement head for token-level keyphrase classification.

    Takes word embeddings from a base encoder and refines them using:
    1. Contextual sentence embeddings (pre/post context from the same document)
    2. Learnable boundary markers (pre_marker, post_marker)
    3. A transformer encoder for cross-attention between words and context
    4. A projection layer for binary classification per token

    The input sequence for each sentence is constructed as:
        [pre_context_sentences] + [PRE_MARKER] + [word_embeddings] + [POST_MARKER] + [post_context_sentences]
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_position_embeddings: int = 768,
    ):
        """Initialize the refinement head.

        Args:
            d_model: Dimension of the model (must match base encoder output).
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network. Defaults to 4 * d_model.
            dropout: Dropout probability.
            max_position_embeddings: Maximum sequence length for positional embeddings.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Projection to single logit per token
        self.projection = nn.Linear(self.d_model, 1)

        # Positional embeddings
        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings, self.d_model
        )

        # Learnable boundary markers
        self.pre_marker = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
        self.post_marker = nn.Parameter(torch.randn(self.d_model), requires_grad=True)

    def forward(
        self,
        word_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_counts: list[int],
    ) -> torch.Tensor:
        """Process word embeddings with document context.

        Args:
            word_embeddings: Word-level embeddings from base encoder,
                shape (num_sentences, max_seq_len, d_model).
            attention_mask: Mask indicating valid tokens,
                shape (num_sentences, max_seq_len).
            sentence_counts: List of sentence counts per document.
                E.g., [3, 5, 2] means 3 documents with 3, 5, and 2 sentences.
                sum(sentence_counts) must equal num_sentences.

        Returns:
            Token-level logits of shape (num_sentences, max_seq_len).
            Logits are placed at positions corresponding to input word embeddings.
        """
        device = word_embeddings.device
        num_sentences, max_seq_len, _ = word_embeddings.shape

        assert num_sentences == sum(sentence_counts), (
            f"Mismatch: word_embeddings has {num_sentences} sentences, "
            f"but sentence_counts sum to {sum(sentence_counts)}"
        )

        # 1. Compute sentence embeddings via mean pooling
        sentence_embs = mean_pooling(word_embeddings, attention_mask)

        # Compute valid token counts per sentence
        sent_len = attention_mask.sum(dim=1)  # (num_sentences,)

        # 2. Build contextual input sequences with markers
        # Pre-context length for each sentence: [0, 1, ..., sc-1] for each document
        pre_context_len = torch.tensor(
            [j for sc in sentence_counts for j in range(sc)],
            dtype=torch.long,
            device=device,
        )

        # Total context length for each sentence (all other sentences in the document)
        total_context_len = torch.tensor(
            [sc for sc in sentence_counts for _ in range(sc)],
            dtype=torch.long,
            device=device,
        )

        # Max input sequence size: (num_sentences - 1) context + max word seq + 2 markers
        max_size = (num_sentences - 1) + int(sent_len.max().item()) + 2

        # Allocate input buffer
        input_seq = torch.zeros(num_sentences, max_size, self.d_model, device=device)

        # Place pre-context sentence embeddings
        for i in range(num_sentences):
            pre_len = pre_context_len[i].item()
            if pre_len > 0:
                input_seq[i, :pre_len] = sentence_embs[:pre_len]

        # Place pre_marker
        input_seq[torch.arange(num_sentences, device=device), pre_context_len] = (
            self.pre_marker
        )

        # Place word embeddings
        word_embs_start = pre_context_len + 1
        post_marker_indices = word_embs_start + sent_len.long()

        for i in range(num_sentences):
            start = word_embs_start[i].item()
            end = post_marker_indices[i].item()
            length = int(sent_len[i].item())
            input_seq[i, start:end] = word_embeddings[i, :length]

        # Place post_marker
        input_seq[torch.arange(num_sentences, device=device), post_marker_indices] = (
            self.post_marker
        )

        # Place post-context sentence embeddings
        post_context_len = total_context_len - pre_context_len - 1
        post_context_start = post_marker_indices + 1

        for i in range(num_sentences):
            pre_len = pre_context_len[i].item()
            post_len = post_context_len[i].item()
            if post_len > 0:
                start = post_context_start[i].item()
                end = start + post_len
                input_seq[i, start:end] = sentence_embs[
                    pre_len + 1 : pre_len + 1 + post_len
                ]

        # 3. Add positional embeddings
        positions = torch.arange(max_size, device=device)
        input_seq = input_seq + self.position_embeddings(positions).unsqueeze(0)

        # 4. Pass through transformer encoder
        output = self.transformer_encoder(input_seq)

        # 5. Project to logits and extract word positions
        output = self.projection(output).squeeze(-1)  # (num_sentences, max_size)

        # Extract logits corresponding to word embedding positions
        masked = torch.zeros(
            num_sentences, max_seq_len, dtype=torch.float32, device=device
        )
        for i in range(num_sentences):
            start = word_embs_start[i].item()
            end = post_marker_indices[i].item()
            length = int(sent_len[i].item())
            masked[i, :length] = output[i, start:end]

        return masked
