"""Refinement head module for keyphrase extraction.

This module implements the custom transformer head that processes word embeddings
with document context to produce token-level keyphrase predictions.
"""

import torch
import torch.nn as nn

from .base_encoder import mean_pooling


class RefinementEncoder(nn.Module):
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
        n_left_ctx: int | None = None,
        n_right_ctx: int | None = None,
    ):
        """Initialize the refinement head.

        Args:
            d_model: Dimension of the model (must match base encoder output).
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network. Defaults to 4 * d_model.
            dropout: Dropout probability.
            max_position_embeddings: Maximum sequence length for positional embeddings.
            n_left_ctx: Maximum number of preceding sentences to use as context.
                If None, use all available preceding sentences in the document.
            n_right_ctx: Maximum number of following sentences to use as context.
                If None, use all available following sentences in the document.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.n_left_ctx = n_left_ctx
        self.n_right_ctx = n_right_ctx

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
        # Pre-context length for each sentence, clamped by n_left_ctx
        pre_context_len = torch.tensor(
            [
                min(j, self.n_left_ctx) if self.n_left_ctx is not None else j
                for sc in sentence_counts
                for j in range(sc)
            ],
            dtype=torch.long,
            device=device,
        )

        # Post-context length for each sentence, clamped by n_right_ctx
        post_context_len = torch.tensor(
            [
                min(sc - j - 1, self.n_right_ctx)
                if self.n_right_ctx is not None
                else (sc - j - 1)
                for sc in sentence_counts
                for j in range(sc)
            ],
            dtype=torch.long,
            device=device,
        )

        # Max input sequence size based on configured context limits
        max_left = (
            self.n_left_ctx
            if self.n_left_ctx is not None
            else (max(sentence_counts) - 1)
        )
        max_right = (
            self.n_right_ctx
            if self.n_right_ctx is not None
            else (max(sentence_counts) - 1)
        )
        max_context = max_left + max_right
        max_size = max_context + int(sent_len.max().item()) + 2  # +2 for markers

        # Allocate input buffer
        input_seq = torch.zeros(num_sentences, max_size, self.d_model, device=device)

        # Compute document offset and sentence index within document for each sentence
        # doc_offset[i] = index of first sentence in document containing sentence i
        # sent_idx[i] = index of sentence i within its document (0-indexed)
        doc_offset = []
        sent_idx = []
        offset = 0
        for sc in sentence_counts:
            for j in range(sc):
                doc_offset.append(offset)
                sent_idx.append(j)
            offset += sc

        # Place pre-context sentence embeddings (immediately preceding sentences)
        for i in range(num_sentences):
            pre_len = pre_context_len[i].item()
            if pre_len > 0:
                # Get the pre_len sentences immediately before sentence i
                # Global index of sentence i within batch = doc_offset[i] + sent_idx[i]
                global_idx = doc_offset[i] + sent_idx[i]
                # Pre-context: sentences [global_idx - pre_len : global_idx]
                input_seq[i, :pre_len] = sentence_embs[
                    global_idx - pre_len : global_idx
                ]

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

        # Place post-context sentence embeddings (immediately following sentences)
        post_context_start = post_marker_indices + 1

        for i in range(num_sentences):
            post_len = post_context_len[i].item()
            if post_len > 0:
                # Global index of sentence i within batch
                global_idx = doc_offset[i] + sent_idx[i]
                # Post-context: sentences [global_idx + 1 : global_idx + 1 + post_len]
                start = post_context_start[i].item()
                end = start + post_len
                input_seq[i, start:end] = sentence_embs[
                    global_idx + 1 : global_idx + 1 + post_len
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
