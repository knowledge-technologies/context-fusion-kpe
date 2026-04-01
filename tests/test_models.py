"""Tests for model modules."""

import torch

from keyword_pecker.models.refinement_encoder import RefinementEncoder


def test_refinement_head_forward_output_shape():
    """Test that RefinementEncoder.forward returns correct output shape and dtype."""
    # Use minimal parameters for lightweight testing
    d_model = 8
    nhead = 2
    num_layers = 1
    max_position_embeddings = 32

    model = RefinementEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=max_position_embeddings,
        num_classes=2,
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

    # Check output shape matches input (num_sentences, max_seq_len) for binary
    assert output.shape == (num_sentences, max_seq_len)
    # Check output dtype is float32
    assert output.dtype == torch.float32


def test_refinement_head_forward_output_shape_iob():
    """Test that RefinementEncoder.forward returns correct output shape for IOB."""
    d_model = 8
    nhead = 2
    num_layers = 1
    max_position_embeddings = 32

    model = RefinementEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=max_position_embeddings,
        num_classes=3,
    )
    model.eval()

    num_sentences = 3
    max_seq_len = 5
    sentence_counts = [2, 1]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.zeros(num_sentences, max_seq_len, dtype=torch.long)
    attention_mask[0, :4] = 1
    attention_mask[1, :3] = 1
    attention_mask[2, :5] = 1

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    # Check output shape for IOB: (num_sentences, max_seq_len, 3)
    assert output.shape == (num_sentences, max_seq_len, 3)
    assert output.dtype == torch.float32


def test_refinement_head_forward_single_document():
    """Test RefinementHead with a single document containing multiple sentences."""
    d_model = 8
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
        num_classes=2,
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
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
        num_classes=2,
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


def test_refinement_head_flexible_context_window():
    """Test RefinementEncoder with limited left/right context windows."""
    d_model = 8
    # Limit to 1 sentence on each side
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=1,
        n_right_ctx=1,
        num_classes=2,
    )
    model.eval()

    # Single document with 5 sentences - only 1 left/right context should be used
    num_sentences = 5
    max_seq_len = 4
    sentence_counts = [5]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    assert output.shape == (num_sentences, max_seq_len)
    assert output.dtype == torch.float32


def test_refinement_head_asymmetric_context():
    """Test RefinementEncoder with asymmetric context windows (different left/right)."""
    d_model = 8
    # More left context than right
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=2,
        n_right_ctx=1,
        num_classes=2,
    )
    model.eval()

    # Two documents with 3 sentences each
    num_sentences = 6
    max_seq_len = 4
    sentence_counts = [3, 3]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    assert output.shape == (num_sentences, max_seq_len)
    assert output.dtype == torch.float32


def test_refinement_head_zero_context():
    """Test RefinementEncoder with zero context (no surrounding sentences)."""
    d_model = 8
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=0,
        n_right_ctx=0,
        num_classes=2,
    )
    model.eval()

    # Single document with 3 sentences - no context should be used
    num_sentences = 3
    max_seq_len = 4
    sentence_counts = [3]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(word_embeddings, attention_mask, sentence_counts)

    assert output.shape == (num_sentences, max_seq_len)
    assert output.dtype == torch.float32


def test_construct_context_semantics():
    """Test that _construct_context places embeddings at correct positions."""
    d_model = 4
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=None,  # unlimited
        n_right_ctx=None,
    )

    # Create deterministic inputs: 1 document with 3 sentences
    # Sentence 0: no pre-context, 2 post-context
    # Sentence 1: 1 pre-context, 1 post-context
    # Sentence 2: 2 pre-context, no post-context
    num_sentences = 3
    max_seq_len = 2
    sentence_counts = [3]

    # Use distinct values so we can verify placement
    word_embeddings = torch.zeros(num_sentences, max_seq_len, d_model)
    word_embeddings[0, 0] = torch.tensor([1.0, 0, 0, 0])  # sent0, word0
    word_embeddings[0, 1] = torch.tensor([2.0, 0, 0, 0])  # sent0, word1
    word_embeddings[1, 0] = torch.tensor([3.0, 0, 0, 0])  # sent1, word0
    word_embeddings[1, 1] = torch.tensor([4.0, 0, 0, 0])  # sent1, word1
    word_embeddings[2, 0] = torch.tensor([5.0, 0, 0, 0])  # sent2, word0
    word_embeddings[2, 1] = torch.tensor([6.0, 0, 0, 0])  # sent2, word1

    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    ctx = model._construct_context(word_embeddings, attention_mask, sentence_counts)

    # Verify context lengths for each sentence
    assert ctx.pre_context_len[0] == 0  # first sentence has no pre-context
    assert ctx.post_context_len[0] == 2  # first sentence has 2 post-context
    assert ctx.pre_context_len[1] == 1  # middle sentence has 1 pre-context
    assert ctx.post_context_len[1] == 1  # middle sentence has 1 post-context
    assert ctx.pre_context_len[2] == 2  # last sentence has 2 pre-context
    assert ctx.post_context_len[2] == 0  # last sentence has no post-context

    # Verify word_embs_start is consistent with pre_context_len
    # word_embs_start = pre_context_len + 1 (for pre_marker)
    assert torch.all(ctx.word_embs_start == ctx.pre_context_len + 1)

    # Verify word_embs_end is consistent with word_embs_start + seq_len
    assert torch.all(ctx.word_embs_end == ctx.word_embs_start + max_seq_len)

    # Verify structure for sentence 1 (middle sentence):
    # Expected layout: [sent0_emb] [PRE_MARKER] [word0] [word1] [POST_MARKER] [sent2_emb]
    #                  idx 0       idx 1        idx 2   idx 3   idx 4         idx 5

    assert ctx.word_embs_start[1] == 2
    assert ctx.word_embs_end[1] == 4

    # Check word embeddings are at correct positions
    assert torch.isclose(ctx.input_seq[1, 2, 0], torch.tensor(3.0))  # word0
    assert torch.isclose(ctx.input_seq[1, 3, 0], torch.tensor(4.0))  # word1

    # Check markers are at correct positions
    assert torch.allclose(ctx.input_seq[1, 1], model.pre_marker)
    assert torch.allclose(ctx.input_seq[1, 4], model.post_marker)

    # Check pre-context (sentence 0 embedding) is at position 0
    # Sentence 0 embedding = mean of word_embeddings[0] = [1.5, 0, 0, 0]
    expected_sent0_emb = word_embeddings[0].mean(dim=0)
    assert torch.allclose(ctx.input_seq[1, 0], expected_sent0_emb)

    # Check post-context (sentence 2 embedding) is at position 5
    expected_sent2_emb = word_embeddings[2].mean(dim=0)
    assert torch.allclose(ctx.input_seq[1, 5], expected_sent2_emb)


def test_construct_context_capped():
    """Test that n_left_ctx and n_right_ctx correctly cap context lengths."""
    d_model = 4
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=1,
        n_right_ctx=1,
    )

    # 1 document with 5 sentences
    num_sentences = 5
    max_seq_len = 2
    sentence_counts = [5]

    word_embeddings = torch.randn(num_sentences, max_seq_len, d_model)
    attention_mask = torch.ones(num_sentences, max_seq_len, dtype=torch.long)

    ctx = model._construct_context(word_embeddings, attention_mask, sentence_counts)

    # Sentence 0: 0 available pre, 4 available post -> capped to (0, 1)
    assert ctx.pre_context_len[0] == 0
    assert ctx.post_context_len[0] == 1

    # Sentence 1: 1 available pre, 3 available post -> capped to (1, 1)
    assert ctx.pre_context_len[1] == 1
    assert ctx.post_context_len[1] == 1

    # Sentence 2 (middle): 2 available pre, 2 available post -> capped to (1, 1)
    assert ctx.pre_context_len[2] == 1
    assert ctx.post_context_len[2] == 1

    # Sentence 3: 3 available pre, 1 available post -> capped to (1, 1)
    assert ctx.pre_context_len[3] == 1
    assert ctx.post_context_len[3] == 1

    # Sentence 4 (last): 4 available pre, 0 available post -> capped to (1, 0)
    assert ctx.pre_context_len[4] == 1
    assert ctx.post_context_len[4] == 0

    # Verify word_embs_start is consistent with pre_context_len
    # word_embs_start = pre_context_len + 1 (for pre_marker)
    assert torch.all(ctx.word_embs_start == ctx.pre_context_len + 1)
