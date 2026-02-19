"""Tests for model modules."""

import torch

from keyword_pecker.models.refinement_head import RefinementHead


def test_refinement_head_forward_output_shape():
    """Test that RefinementHead.forward returns correct output shape and dtype."""
    # Use minimal parameters for lightweight testing
    d_model = 8
    nhead = 2
    num_layers = 1
    max_position_embeddings = 32

    model = RefinementHead(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=max_position_embeddings,
    )
    model.eval()

    # Create mock inputs: 2 documents with 2 and 1 sentences respectively
    num_sentences = 3
    max_seq_len = 5
    sentence_counts = [2, 1]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    # Variable-length sequences: lengths 4, 3, 5
    attention_mask = torch.zeros(num_sentences, max_seq_len, dtype=torch.long)
    attention_mask[0, :4] = 1
    attention_mask[1, :3] = 1
    attention_mask[2, :5] = 1

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    # Check output shape matches input (num_sentences, max_seq_len)
    assert output.shape == (num_sentences, max_seq_len)
    # Check output dtype is float32
    assert output.dtype == torch.float32


def test_refinement_head_forward_single_document():
    """Test RefinementHead with a single document containing multiple sentences."""
    d_model = 8
    model = RefinementHead(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
    )
    model.eval()

    # Single document with 3 sentences
    num_sentences = 3
    max_seq_len = 4
    sentence_counts = [3]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    assert output.shape == (num_sentences, max_seq_len)
    assert output.dtype == torch.float32


def test_refinement_head_forward_single_sentence():
    """Test RefinementHead with a single sentence (edge case)."""
    d_model = 8
    model = RefinementHead(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
    )
    model.eval()

    # Single document with 1 sentence
    num_sentences = 1
    max_seq_len = 6
    sentence_counts = [1]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    assert output.shape == (num_sentences, max_seq_len)
    assert output.dtype == torch.float32
