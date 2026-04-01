"""
PyTorch Dataset and DataLoader utilities for keyphrase extraction.

This module provides:
- KeyphraseDataset: A Dataset wrapping a Polars DataFrame of tokenized sentences
- DocumentBatchSampler: A BatchSampler that groups sentences by document and
  supports two-level batching (outer chunk for gradient accumulation, inner batch)
- keyphrase_collate_fn: Collate function that pads sequences and computes sentence_counts
- create_keyphrase_dataloader: Factory function to create a DataLoader with all components
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class KeyphraseDataset(Dataset):
    """Dataset for keyphrase extraction at sentence level.

    Each item represents a single sentence with its token IDs and keyword mask.
    The DataFrame must be sorted by (doc_id, doc_row, sent_num) to preserve document order.

    Attributes:
        df: Source Polars DataFrame with sentence data
    """

    def __init__(self, df: pl.DataFrame):
        """
        Args:
            df: Polars DataFrame with columns [doc_id, doc_row, sent_num, token_ids, keyword_mask]
                Must be sorted by (doc_id, doc_row, sent_num).
        """
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sentence sample.

        Returns:
            dict with keys:
                - doc_id: Document file identifier
                - doc_row: Row index within the document file
                - token_ids: List[int] of token IDs
                - keyword_mask: List[int] of keyword labels per token
                  - IOB scheme: 0 (Outside), 1 (Beginning), 2 (Inside)
                  - Binary scheme: 0 (not keyword), 1 (keyword)
        """
        row = self.df.row(idx, named=True)
        return {
            "doc_id": row["doc_id"],
            "doc_row": row["doc_row"],
            "token_ids": row["token_ids"],
            "keyword_mask": row["keyword_mask"],
            "keywords": row["keywords"],
        }


class DocumentBatchSampler(Sampler[list[int]]):
    """Batch sampler that respects document boundaries and supports two-level batching.

    This sampler implements the batching strategy from the original training loop:
    1. Shuffles at document level (groups of sentences with same doc_id+doc_row)
    2. Yields outer chunks of `outer_batch_size` sentences (for gradient accumulation)
    3. Each outer chunk is further split into inner batches of `inner_batch_size`

    The inner batches handle document boundary splitting: when a document spans
    a batch boundary, the sentence_counts are split accordingly.

    Attributes:
        dataset: The KeyphraseDataset to sample from
        outer_batch_size: Number of sentences per outer chunk (gradient accumulation unit)
        inner_batch_size: Number of sentences per inner batch (forward pass unit)
        shuffle: Whether to shuffle documents each epoch
        drop_last: Whether to drop the last incomplete outer batch
    """

    def __init__(
        self,
        dataset: KeyphraseDataset,
        outer_batch_size: int = 140,
        inner_batch_size: int = 70,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.outer_batch_size = outer_batch_size
        self.inner_batch_size = inner_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Precompute document boundaries (list of (start_idx, end_idx) per document)
        self._doc_boundaries = self._compute_doc_boundaries()

    def _compute_doc_boundaries(self) -> list[tuple[int, int]]:
        """Compute start and end indices for each document in the dataset.

        A document is uniquely identified by (doc_id, doc_row) combination.
        """
        df = self.dataset.df
        boundaries = []

        if len(df) == 0:
            return boundaries

        # Get doc_id and doc_row columns
        doc_ids = df["doc_id"].to_list()
        doc_rows = df["doc_row"].to_list()

        current_doc = (doc_ids[0], doc_rows[0])
        start_idx = 0

        for idx in range(1, len(df)):
            new_doc = (doc_ids[idx], doc_rows[idx])
            if new_doc != current_doc:
                boundaries.append((start_idx, idx))
                start_idx = idx
                current_doc = new_doc

        # Don't forget the last document
        boundaries.append((start_idx, len(df)))

        return boundaries

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices.

        Each yielded batch is a list of sentence indices forming one inner batch.
        Inner batches are yielded sequentially within each outer chunk.
        """
        # Get document order (shuffle at document level if requested)
        doc_indices = list(range(len(self._doc_boundaries)))
        if self.shuffle:
            random.shuffle(doc_indices)

        # Flatten to sentence indices in document-shuffled order
        all_indices = []
        for doc_idx in doc_indices:
            start, end = self._doc_boundaries[doc_idx]
            all_indices.extend(range(start, end))

        # Process in outer chunks (gradient accumulation units)
        num_outer_batches = math.ceil(len(all_indices) / self.outer_batch_size)
        if self.drop_last:
            num_outer_batches = len(all_indices) // self.outer_batch_size

        for outer_idx in range(num_outer_batches):
            outer_start = outer_idx * self.outer_batch_size
            outer_end = min(outer_start + self.outer_batch_size, len(all_indices))
            outer_chunk_indices = all_indices[outer_start:outer_end]

            # Split outer chunk into inner batches
            num_inner_batches = math.ceil(
                len(outer_chunk_indices) / self.inner_batch_size
            )
            for inner_idx in range(num_inner_batches):
                inner_start = inner_idx * self.inner_batch_size
                inner_end = min(
                    inner_start + self.inner_batch_size, len(outer_chunk_indices)
                )
                yield outer_chunk_indices[inner_start:inner_end]

    def __len__(self) -> int:
        """Return total number of inner batches."""
        total_sentences = len(self.dataset)
        if self.drop_last:
            num_outer = total_sentences // self.outer_batch_size
        else:
            num_outer = math.ceil(total_sentences / self.outer_batch_size)

        # Each outer chunk yields ceil(outer_batch_size / inner_batch_size) inner batches
        # except possibly the last outer chunk
        inner_per_outer = math.ceil(self.outer_batch_size / self.inner_batch_size)

        if self.drop_last or total_sentences % self.outer_batch_size == 0:
            return num_outer * inner_per_outer
        else:
            # Last outer chunk may have fewer inner batches
            last_outer_size = total_sentences % self.outer_batch_size
            last_inner_count = math.ceil(last_outer_size / self.inner_batch_size)
            return (num_outer - 1) * inner_per_outer + last_inner_count


def keyphrase_collate_fn(batch: list[dict]) -> dict:
    """Collate function that pads sequences and computes sentence_counts.

    This function replicates the tensor preparation logic from the original
    training loop (lines 228-269 of train-model.py).

    Args:
        batch: List of samples from KeyphraseDataset.__getitem__

    Returns:
        dict with keys:
            - input_ids: Padded token IDs tensor (batch_size, max_len)
            - attention_mask: Padding mask tensor (batch_size, max_len)
            - target: Padded keyword mask tensor (batch_size, max_len)
            - sentence_counts: List[int] of sentences per document in this batch
            - doc_ids: List[str] of document IDs
            - doc_rows: List[int] of document row indices
            - keywords: List[List[str]] of gold keywords per sample
    """
    num_rows = len(batch)

    # Compute max sequence length in this batch
    max_len = max(len(sample["token_ids"]) for sample in batch)

    # Initialize padded tensors
    input_ids = torch.zeros((num_rows, max_len), dtype=torch.long)
    attention_mask = torch.zeros((num_rows, max_len), dtype=torch.long)
    target = torch.zeros((num_rows, max_len), dtype=torch.long)

    # Fill tensors row by row
    for i, sample in enumerate(batch):
        tokens = sample["token_ids"]
        kw_mask = sample["keyword_mask"]
        seq_len = len(tokens)

        input_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        target[i, :seq_len] = torch.tensor(kw_mask, dtype=torch.long)

    # Compute sentence_counts from (doc_id, doc_row) groupings
    # This counts consecutive sentences with the same document identity
    sentence_counts = []
    if num_rows > 0:
        current_doc = (batch[0]["doc_id"], batch[0]["doc_row"])
        current_count = 1

        for i in range(1, num_rows):
            new_doc = (batch[i]["doc_id"], batch[i]["doc_row"])
            if new_doc == current_doc:
                current_count += 1
            else:
                sentence_counts.append(current_count)
                current_doc = new_doc
                current_count = 1

        sentence_counts.append(current_count)

    # Extract document identifiers and keywords for each sample
    doc_ids = [sample["doc_id"] for sample in batch]
    doc_rows = [sample["doc_row"] for sample in batch]
    keywords = [sample["keywords"] for sample in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": target,
        "sentence_counts": sentence_counts,
        "doc_ids": doc_ids,
        "doc_rows": doc_rows,
        "keywords": keywords,
    }


def create_keyphrase_dataloader(
    df: pl.DataFrame,
    outer_batch_size: int = 140,
    inner_batch_size: int = 70,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for keyphrase extraction training.

    This factory function assembles all components:
    - KeyphraseDataset wrapping the Polars DataFrame
    - DocumentBatchSampler for document-aware two-level batching
    - keyphrase_collate_fn for padding and sentence_counts

    Args:
        df: Polars DataFrame with columns [doc_id, doc_row, sent_num, token_ids, keyword_mask]
            Must be sorted by (doc_id, doc_row, sent_num).
        outer_batch_size: Sentences per gradient accumulation unit (default: 140)
        inner_batch_size: Sentences per forward pass (default: 70)
        shuffle: Whether to shuffle at document level each epoch (default: True)
        drop_last: Whether to drop incomplete outer batches (default: False)
        num_workers: DataLoader workers (default: 0)

    Returns:
        DataLoader yielding batches with keys:
            - input_ids: (batch_size, max_len) long tensor
            - attention_mask: (batch_size, max_len) long tensor
            - target: (batch_size, max_len) float tensor
            - sentence_counts: List[int]

    Example:
        >>> import polars as pl
        >>> train_df = pl.read_parquet("data/train.parquet")
        >>> train_loader = create_keyphrase_dataloader(train_df, shuffle=True)
        >>> for batch in train_loader:
        ...     input_ids = batch["input_ids"]
        ...     attention_mask = batch["attention_mask"]
        ...     target = batch["target"]
        ...     sentence_counts = batch["sentence_counts"]
        ...     # Forward pass...
    """
    dataset = KeyphraseDataset(df)

    batch_sampler = DocumentBatchSampler(
        dataset=dataset,
        outer_batch_size=outer_batch_size,
        inner_batch_size=inner_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=keyphrase_collate_fn,
        num_workers=num_workers,
    )

    return dataloader
