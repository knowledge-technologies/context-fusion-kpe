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


def test_construct_context_unequal_lengths():
    """Test _construct_context with unequal sentence lengths and multiple documents.

    This test verifies that padding masks are correctly applied when sentences
    have different token counts and documents have different sentence counts.

    Setup:
    - 2 documents: doc0 has 3 sentences, doc1 has 1 sentence
    - Sentence token lengths: 2, 4, 1, 3 (unequal)

    Layout per sentence: [pre_context] [PRE_MARKER] [words] [POST_MARKER] [post_context] [PADDING]
    """
    d_model = 4
    model = RefinementEncoder(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        max_position_embeddings=32,
        n_left_ctx=None,  # unlimited context
        n_right_ctx=None,
    )

    num_sentences = 4
    max_seq_len = 4
    sentence_counts = [3, 1]  # doc0: 3 sents, doc1: 1 sent

    # Create deterministic word embeddings with unique values per token
    # Value = sentence_index * 100 + token_index + 1 (e.g., sent2 tok0 = 201)
    word_embeddings = torch.zeros(num_sentences, max_seq_len, d_model)
    for sent_idx in range(num_sentences):
        for tok_idx in range(max_seq_len):
            word_embeddings[sent_idx, tok_idx, 0] = sent_idx * 100 + tok_idx + 1

    # Attention mask reflecting unequal sentence lengths
    attention_mask = torch.zeros(num_sentences, max_seq_len, dtype=torch.long)
    attention_mask[0, :2] = 1  # sent0: 2 tokens
    attention_mask[1, :4] = 1  # sent1: 4 tokens (longest)
    attention_mask[2, :1] = 1  # sent2: 1 token
    attention_mask[3, :3] = 1  # sent3: 3 tokens

    ctx = model._construct_context(word_embeddings, attention_mask, sentence_counts)

    # --- Verify context lengths ---
    # doc0 has 3 sentences: max context = 3 - 1 = 2 total (pre + post)
    # sent0: first in doc0 -> 0 pre, 2 post (sent1, sent2)
    assert ctx.pre_context_len[0] == 0
    assert ctx.post_context_len[0] == 2
    # sent1: middle of doc0 -> 1 pre (sent0), 1 post (sent2)
    assert ctx.pre_context_len[1] == 1
    assert ctx.post_context_len[1] == 1
    # sent2: last in doc0 -> 2 pre (sent0, sent1), 0 post
    assert ctx.pre_context_len[2] == 2
    assert ctx.post_context_len[2] == 0
    # doc1 has 1 sentence: no context available at all
    # sent3: only sentence in doc1 -> 0 pre, 0 post
    assert ctx.pre_context_len[3] == 0
    assert ctx.post_context_len[3] == 0

    # --- Verify max_size ---
    # max_context = max(sentence_counts) - 1 = 3 - 1 = 2
    # max_sent_len = 4 (from sent1)
    # max_size = 2 + 4 + 2(markers) = 8
    assert ctx.input_seq.shape[1] == 8

    # --- Verify padding mask ---
    # Valid length per sentence = pre + 1(pre_marker) + sent_len + 1(post_marker) + post
    # sent0: 0 + 1 + 2 + 1 + 2 = 6 valid -> positions 6,7 are padding
    # sent1: 1 + 1 + 4 + 1 + 1 = 8 valid -> no padding
    # sent2: 2 + 1 + 1 + 1 + 0 = 5 valid -> positions 5,6,7 are padding
    # sent3: 0 + 1 + 3 + 1 + 0 = 5 valid -> positions 5,6,7 are padding
    assert ctx.padding_mask.shape == (num_sentences, 8)

    # sent0: first 6 positions valid, last 2 are padding
    assert not ctx.padding_mask[0, :6].any(), "sent0: positions 0-5 should be valid"
    assert ctx.padding_mask[0, 6:].all(), "sent0: positions 6-7 should be padding"

    # sent1: all 8 positions valid (longest sentence fills entire buffer)
    assert not ctx.padding_mask[1].any(), "sent1: all positions should be valid"

    # sent2: first 5 valid, last 3 padding
    assert not ctx.padding_mask[2, :5].any(), "sent2: positions 0-4 should be valid"
    assert ctx.padding_mask[2, 5:].all(), "sent2: positions 5-7 should be padding"

    # sent3: first 5 valid, last 3 padding
    assert not ctx.padding_mask[3, :5].any(), "sent3: positions 0-4 should be valid"
    assert ctx.padding_mask[3, 5:].all(), "sent3: positions 5-7 should be padding"

    # --- Verify word embedding placement for sent1 (middle sentence) ---
    # Layout: [sent0_emb] [PRE] [w0] [w1] [w2] [w3] [POST] [sent2_emb]
    #          pos 0      pos1  pos2  pos3  pos4  pos5  pos6    pos7
    assert ctx.word_embs_start[1] == 2  # words start after pre_context(1) + marker(1)
    assert ctx.word_embs_end[1] == 6  # words end before post_marker
    for tok_idx in range(4):
        expected_val = 100 + tok_idx + 1  # sent1: 1*100 + tok_idx + 1
        assert torch.isclose(
            ctx.input_seq[1, 2 + tok_idx, 0],
            torch.tensor(float(expected_val)),
        ), f"sent1 token {tok_idx} should be at position {2 + tok_idx}"

    # --- Verify marker placement for sent1 ---
    # pre_marker at position pre_context_len = 1
    assert torch.allclose(ctx.input_seq[1, 1], model.pre_marker)
    # post_marker at position pre + 1 + sent_len = 1 + 1 + 4 = 6
    assert torch.allclose(ctx.input_seq[1, 6], model.post_marker)

    # --- Verify pre-context for sent1 ---
    # pre-context should be sentence 0's mean-pooled embedding
    # sent0 has 2 valid tokens with values 1 and 2 -> mean = 1.5
    expected_sent0_emb = word_embeddings[0, :2].mean(dim=0)
    assert torch.allclose(ctx.input_seq[1, 0], expected_sent0_emb), (
        "sent1 pre-context should be sent0's mean-pooled embedding"
    )

    # --- Verify post-context for sent1 ---
    # post-context should be sentence 2's mean-pooled embedding
    # sent2 has 1 valid token with value 201 -> mean = 201
    expected_sent2_emb = word_embeddings[2, :1].mean(dim=0)
    assert torch.allclose(ctx.input_seq[1, 7], expected_sent2_emb), (
        "sent1 post-context should be sent2's mean-pooled embedding"
    )

    # --- Verify doc boundary isolation ---
    # sent3 is the only sentence in doc1, so it has no context
    # Layout: [PRE] [w0] [w1] [w2] [POST] [PAD] [PAD] [PAD]
    #          pos0  pos1  pos2  pos3  pos4   pos5   pos6   pos7
    assert ctx.word_embs_start[3] == 1  # words start after pre_marker
    assert ctx.word_embs_end[3] == 4  # words end after 3 tokens
    assert torch.allclose(ctx.input_seq[3, 0], model.pre_marker)
    assert torch.allclose(ctx.input_seq[3, 4], model.post_marker)
    # Positions 5-7 should be zero (padding, not part of any context)
    assert (ctx.input_seq[3, 5:] == 0).all(), "sent3 padding positions should be zero"
