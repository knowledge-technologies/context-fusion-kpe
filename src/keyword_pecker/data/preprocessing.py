"""Data preprocessing functions for keyphrase extraction.

This module contains all logic for processing raw JSON datasets into
training-ready Parquet files with token-level keyword labels.

The preprocessing pipeline:
1. Load raw JSON files (title, abstract, fulltext, keywords)
2. Segment text into sentences using spaCy
3. Tokenize sentences using HuggingFace tokenizer
4. Match keywords to token spans using stemmed regex
5. Create binary keyword masks for each token
6. Write results to Parquet files

All functions are stateless and accept their dependencies as arguments.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import regex
import spacy
import torch
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from keyword_pecker.constants import IGNORE_IDX
from keyword_pecker.utils import get_logger

logger = get_logger("preprocessing")

# ============================================================================
# Schema Definition
# ============================================================================

SENTENCE_SCHEMA = {
    "doc_id": pl.Utf8,
    "doc_row": pl.Int32,
    "sent_num": pl.Int32,
    "sent_txt": pl.Utf8,
    "keywords": pl.List(pl.Utf8),
    "sent_keywords": pl.List(pl.Utf8),
    "token_ids": pl.List(pl.Int32),
    "attention_mask": pl.List(pl.Int32),
    "keyword_mask": pl.List(pl.Int32),
    "word_ids": pl.List(pl.Int64),
    "word_list": pl.List(pl.Utf8),
}

KEYWORD_MASK_LABELS = {
    "binary": {0: "not_keyword", 1: "keyword"},
    "IOB": {0: "Outside", 1: "Beginning", 2: "Inside"},
}


# ============================================================================
# Sentence Segmentation
# ============================================================================


def load_spacy_model(
    model_name: str = "en_core_web_sm",
    exclude: list[str] | None = None,
) -> spacy.Language:
    """Load spaCy model for sentence segmentation.

    Args:
        model_name: SpaCy model name.
        exclude: Pipeline components to exclude. Defaults to excluding
            NER, lemmatizer, and attribute_ruler for efficiency.

    Returns:
        Loaded spaCy model.
    """
    if exclude is None:
        exclude = ["ner", "lemmatizer", "attribute_ruler"]
    return spacy.load(model_name, exclude=exclude)


def segment_sentences(
    texts: list[str],
    nlp: spacy.Language,
    batch_size: int = 50,
    n_process: int = 1,
) -> list[spacy.tokens.Doc]:
    """Segment texts into sentences using spaCy.

    Uses nlp.pipe() for efficient batch processing.

    Args:
        texts: List of texts to segment.
        nlp: Loaded spaCy model.
        batch_size: Batch size for processing.
        n_process: Number of processes for parallel processing.

    Returns:
        List of spaCy Doc objects with sentence boundaries.
    """
    return list(nlp.pipe(texts, batch_size=batch_size, n_process=n_process))


# ============================================================================
# Keyword Matching
# ============================================================================


def _normalize_text(text: str) -> str:
    """Apply legacy-style text normalization before keyword matching.

    Replaces hyphens, slashes, and asterisks with spaces so that
    multi-word keywords like "deep-learning" match "deep learning" in text.
    Matches the normalization in legacy preprocess_text().

    Args:
        text: Raw sentence text.

    Returns:
        Normalized text with -, /, ∗ replaced by spaces.
    """
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("∗", " ")
    return text


def build_keyword_regex(
    keywords: list[str],
    stemmer: PorterStemmer,
) -> regex.Pattern:
    """Build regex pattern from keywords with word boundaries.

    Creates a compiled regex that matches keywords as complete words,
    sorted by length (longest first) to prioritize longer matches.

    Args:
        keywords: List of keyword phrases.
        stemmer: Porter stemmer instance.

    Returns:
        Compiled regex pattern for matching keywords.

    Example:
        >>> stemmer = PorterStemmer()
        >>> pattern = build_keyword_regex(["machine learning"], stemmer)
        >>> pattern.pattern
        '(?i)\\\\bmachine\\\\s+learning\\\\b'
    """
    regex_parts = []

    for kw in keywords:
        word_re_parts = []
        for w in kw.split():
            word_re_parts.append(rf"\b{regex.escape(w)}\b")
        regex_parts.append(r"\s+".join(word_re_parts))

    # Sort by length (longest first) to prioritize longer matches
    combined_pattern = "|".join(sorted(regex_parts, key=len, reverse=True))
    return regex.compile(combined_pattern, flags=regex.IGNORECASE)


def resolve_keyword_spans(
    sentence: str,
    offsets: np.ndarray,
    num_total_tokens: int,
    regex_pattern: regex.Pattern,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Resolve keyword spans in sentence to token ranges.

    This is the core logic from the original f_resolve_kwds function,
    preserved to maintain exact behavior.

    Maps keyword occurrences in text to token index ranges using
    character offsets from the tokenizer.

    Args:
        sentence: Lowercased sentence text.
        offsets: Token offset mapping from tokenizer (shape: [seq_len, 2]).
        num_total_tokens: Number of tokens (excluding padding).
        regex_pattern: Compiled regex pattern for matching keywords.

    Returns:
        Tuple of (matched_keywords, token_ranges):
        - matched_keywords: List of keyword strings found in sentence.
        - token_ranges: List of (start_tok, end_tok) tuples.

    Note:
        Handles overlapping keywords by skipping subsequent matches.
        Handles truncation by stopping when keyword extends beyond tokens.
    """
    kwds, word_token_slice_ixs = [], []

    last_token_ends_at = offsets[num_total_tokens - 2, 1]
    token_boundaries = np.zeros(len(sentence) + 1, dtype=np.int64)
    token_boundaries[offsets[1:-1, 0]] = 1
    token_boundaries = np.cumsum(token_boundaries)

    covered_spans = np.zeros(len(sentence) + 1)

    for m in regex_pattern.finditer(sentence, overlapped=True):
        w_string = m.group(0)
        w_start, w_end = m.start(), m.start() + len(w_string)
        assert w_end == m.end()

        if covered_spans[w_start:w_end].sum() > 0:
            continue

        if w_end > last_token_ends_at:
            break

        kwds.append(w_string)

        assert sentence[w_start:w_end] == w_string, (
            sentence[w_start:w_end],
            w_string,
        )
        word_token_slice_ixs.append(
            (
                token_boundaries[w_start].item(),
                token_boundaries[w_end - 1].item() + 1,
            )
        )
        covered_spans[w_start:w_end] = 1

    return kwds, word_token_slice_ixs


def resolve_keyword_spans_stem(
    sentence: str,
    offsets: np.ndarray,
    num_total_tokens: int,
    keywords: list[str],
    stemmer: PorterStemmer,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Resolve keyword spans using stem-equality matching (legacy gold-standard approach).

    Instead of regex matching, this function uses word-level stem equality:
    for each position in the document, it checks whether the stems of the
    document words match the stems of the keyword words exactly. This is the
    same algorithm used in the legacy preprocessing.py tokenize_doc function.

    The key advantage over regex is symmetry: stem("aims") == "aim" which is
    not equal to stem("ai") == "ai", so keyword "ai" will NOT match "aims".
    A stem-prefix regex \\b{stem}\\w*\\b would incorrectly match "aims" for
    keyword "ai" because "aims" starts with "ai".

    Character offsets from the matched word spans are looked up in the same
    token_boundaries array as resolve_keyword_spans, so the token index
    mapping is identical.

    Args:
        sentence: Lowercased sentence text.
        offsets: Token offset mapping from tokenizer (shape: [seq_len, 2]).
        num_total_tokens: Number of tokens (excluding padding).
        keywords: List of keyword phrases (lowercased).
        stemmer: Porter stemmer instance.

    Returns:
        Tuple of (matched_keywords, token_ranges):
        - matched_keywords: List of keyword strings found in sentence.
        - token_ranges: List of (start_tok, end_tok) tuples.

    Note:
        Handles overlapping keywords by skipping subsequent matches.
        Handles truncation by stopping when keyword extends beyond tokens.
    """
    kwds, word_token_slice_ixs = [], []

    # Build token_boundaries array (identical to resolve_keyword_spans)
    last_token_ends_at = offsets[num_total_tokens - 2, 1]
    token_boundaries = np.zeros(len(sentence) + 1, dtype=np.int64)
    token_boundaries[offsets[1:-1, 0]] = 1
    token_boundaries = np.cumsum(token_boundaries)

    covered_spans = np.zeros(len(sentence) + 1)

    # Extract word spans from sentence with their character positions.
    # \b\w+\b ensures we match complete words only (same word-boundary
    # semantics as the legacy word_tokenize approach, without NLTK dependency).
    word_spans: list[tuple[str, int, int]] = [
        (m.group(0), m.start(), m.end()) for m in regex.finditer(r"\b\w+\b", sentence)
    ]
    stemmed_doc_words = [stemmer.stem(w) for w, _, _ in word_spans]

    # Pre-stem all keyword phrases once
    stemmed_keywords: list[list[str]] = [
        [stemmer.stem(w) for w in kw.split()] for kw in keywords
    ]

    n_words = len(word_spans)

    # Sort by keyword length descending (longest-first), same priority as
    # the regex approach, to prefer longer matches over shorter overlapping ones
    kw_order = sorted(
        range(len(keywords)),
        key=lambda i: len(stemmed_keywords[i]),
        reverse=True,
    )

    for kw_idx in kw_order:
        stemmed_kw = stemmed_keywords[kw_idx]
        kw_len = len(stemmed_kw)

        for j in range(n_words - kw_len + 1):
            # Stem-equality check: every word in the keyword must match
            if any(stemmed_doc_words[j + k] != stemmed_kw[k] for k in range(kw_len)):
                continue

            # Found a stem-equal match; get character span
            _, char_start, _ = word_spans[j]
            _, _, char_end = word_spans[j + kw_len - 1]

            if covered_spans[char_start:char_end].sum() > 0:
                # Overlapping with an already-matched keyword; skip
                continue

            if char_end > last_token_ends_at:
                # Sentence was truncated; keyword extends beyond tokenized region
                continue

            # The matched surface form is the original sentence text for the span
            w_string = sentence[char_start:char_end]
            kwds.append(w_string)
            word_token_slice_ixs.append(
                (
                    token_boundaries[char_start].item(),
                    token_boundaries[char_end - 1].item() + 1,
                )
            )
            covered_spans[char_start:char_end] = 1

    return kwds, word_token_slice_ixs


# ============================================================================
# Tokenization & Keyword Alignment
# ============================================================================


def extract_word_level_data(
    batch_tokenized: BatchEncoding,
    batch_sentence_strings: list[str],
) -> tuple[list[list[int]], list[list[str]]]:
    """Extract word_ids and word_lists from a tokenized batch.

    Builds word_lists from offset_mapping (not sentence.split()) to handle
    tokenizers that assign separate word_ids to punctuation or hyphens.
    E.g. "state-of-the-art" gets 8 word_ids but split() gives 1 element.

    Args:
        batch_tokenized: Tokenizer output (BatchEncoding) with word_ids()
            and offset_mapping available.
        batch_sentence_strings: Original sentence texts, one per batch item.

    Returns:
        Tuple of (word_ids_per_sent, word_lists_per_sent):
        - word_ids_per_sent: List of word_id lists, one per sentence.
          Each list maps tokens to word indices (-1 for special/padding).
        - word_lists_per_sent: List of word lists, one per sentence.
          Each list is indexed by word_id.
    """
    word_ids_per_sent = []
    word_lists_per_sent = []

    for s_idx in range(len(batch_sentence_strings)):
        wid_list = batch_tokenized.word_ids(batch_index=s_idx)
        word_ids_per_sent.append([w if w is not None else -1 for w in wid_list])

        offsets = batch_tokenized["offset_mapping"][s_idx]
        word_offsets: dict[int, list[tuple[int, int]]] = {}
        for tok_idx, wid in enumerate(wid_list):
            if wid is not None:
                word_offsets.setdefault(wid, []).append(offsets[tok_idx])
        word_map = {}
        for wid, offs in word_offsets.items():
            char_start = min(o[0] for o in offs)
            char_end = max(o[1] for o in offs)
            word_map[wid] = batch_sentence_strings[s_idx][char_start:char_end]
        word_lists_per_sent.append([word_map[i] for i in range(len(word_map))])

    return word_ids_per_sent, word_lists_per_sent


def tokenize_sentences(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    truncation: bool = True,
    padding: bool = True,
) -> dict[str, Any]:
    """Tokenize sentences with offset mapping.

    Args:
        sentences: List of sentence texts.
        tokenizer: HuggingFace tokenizer.
        truncation: Whether to truncate sequences exceeding model_max_length.
        padding: Whether to pad sequences to longest in batch.

    Returns:
        Tokenizer output dict with input_ids, attention_mask, offset_mapping.
    """
    return tokenizer(
        sentences,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=truncation,
        padding=padding,
    )


def create_keyword_mask(
    keyword_token_ranges: list[tuple[int, int]],
    sent_tok_len: int,
    scheme: str = "IOB",
    device: str = "cpu",
    word_ids: list[int] | None = None,
) -> torch.Tensor:
    """Create keyword mask for tokens using binary or IOB labeling scheme.

    When ``word_ids`` is provided, only the first subtoken of each word receives
    a label; continuation subtokens are set to ``IGNORE_IDX`` (PyTorch ignore index).
    This matches the legacy ``tokenize_and_align_labels`` behavior.

    Args:
        keyword_token_ranges: List of [start, end) token index pairs (`end` is not included).
        sent_tok_len: Total number of tokens in sentence.
        scheme: Labeling scheme - "IOB" (0=Outside, 1=Beginning, 2=Inside) or
            "binary" (0=not keyword, 1=keyword). Defaults to "IOB".
        device: Torch device for computation.
        word_ids: Optional list mapping each token to its word index (-1 for
            special/padding tokens). When provided, non-initial subtokens get IGNORE_IDX.

    Returns:
        Tensor of shape (sent_tok_len,) with labels:
        - IOB scheme: 0 (Outside), 1 (Beginning), 2 (Inside), IGNORE_IDX (ignored subtoken)
        - Binary scheme: 0 (not keyword), 1 (keyword), IGNORE_IDX (ignored subtoken)

    Raises:
        ValueError: If overlapping keywords detected (should not happen
            with the skip strategy in resolve_keyword_spans).
    """

    # if no keyphrases in sentece
    if not keyword_token_ranges:
        # if we have word ids, sort those to be ignored
        if word_ids is not None:
            targets = torch.full((sent_tok_len,), IGNORE_IDX, dtype=torch.long)
            for t in range(sent_tok_len):
                if word_ids[t] == -1:
                    targets[t] = IGNORE_IDX
                elif t == 0 or word_ids[t] != word_ids[t - 1]:
                    targets[t] = 0
            return targets
        return torch.zeros(sent_tok_len, dtype=torch.long)

    # if we have keyphrases, use token ranges
    tok_start, tok_end = zip(*keyword_token_ranges)
    tok_start = torch.tensor(tok_start)
    tok_end = torch.tensor(tok_end)

    # create array of indices (one sequence for each keyphrase)
    arange = torch.arange(sent_tok_len).tile(len(keyword_token_ranges), 1)

    def _binary_labels(idxs, starts, ends):

        # pre-allocate all 1's
        targets = torch.ones_like(idxs, device=device)

        # no zero-out all tokens before and after keyword
        # tokens for each keyword
        targets[idxs < starts.view(-1, 1)] = 0
        targets[idxs >= ends.view(-1, 1)] = 0

        # sum across dim 0, to obtain a
        # single array of 0's and 1's
        targets = targets.sum(dim=0)

        return targets

    def _iob_labels(idxs, starts, ends):

        targets = torch.zeros_like(idxs, device=device)

        begin_phrase = idxs == starts.view(-1, 1)
        targets[begin_phrase] = 1

        within_phrase = (idxs > starts.view(-1, 1)) & (idxs < ends.view(-1, 1))
        targets[within_phrase] = 2

        return targets.sum(dim=0)

    factory = _iob_labels if scheme == "IOB" else _binary_labels

    # place IOB labels (0, 1, 2)
    targets = factory(idxs=arange, starts=tok_start, ends=tok_end)

    if scheme == "binary":
        if targets.max() > 1:
            raise ValueError(
                f"Overlapping keywords detected: max overlap = {targets.max()}"
            )
    else:
        if targets.max() > 2:
            raise ValueError(
                f"Overlapping keywords detected: max overlap = {targets.max()}"
            )

    if word_ids is not None:
        prev_wid = None
        for t in range(sent_tok_len):
            wid = word_ids[t]
            if wid == -1:
                targets[t] = IGNORE_IDX
            elif wid != prev_wid:
                pass  # first subtoken of word → keep label
            else:
                targets[t] = IGNORE_IDX  # continuation subtoken → ignore
            prev_wid = wid

    return targets


# ============================================================================
# Parquet I/O
# ============================================================================


def write_parquet(
    data_rows: list[dict],
    output_path: Path,
    schema: dict | None = None,
) -> int:
    """Write processed sentence data to Parquet file.

    Args:
        data_rows: List of sentence data dictionaries.
        output_path: Path to output .parquet file.
        schema: Optional Polars schema (defaults to SENTENCE_SCHEMA).

    Returns:
        Number of rows written.
    """
    if not data_rows:
        return 0

    lf = pl.LazyFrame(data_rows, schema=schema or SENTENCE_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(output_path, compression="snappy")

    n_rows = lf.select(pl.len()).collect().item()

    return n_rows


# ============================================================================
# High-level Pipeline Functions
# ============================================================================


def extract_keywords(row: Any, columns: frozenset[str]) -> list[str] | None:
    """Extract and normalize keywords from a data row.

    Args:
        row: DataFrame row (namedtuple from itertuples).
        columns: Set of column names in the DataFrame.

    Returns:
        List of lowercase, stripped keywords, or None if no valid keywords.
    """
    if "keyword" in columns:
        kws = row["keyword"]
    elif "keywords" in columns:
        kws = row["keywords"]
    else:
        return None

    if isinstance(kws, str):
        kws = kws.split(";")

    kws = [kw.lower().strip() for kw in kws if kw.strip()]
    return kws if kws else None


def collect_sentences(
    row: Any,
    nlp_abstract: spacy.tokens.Doc | None,
    nlp_fulltext: spacy.tokens.Doc | None,
    columns: frozenset[str],
) -> list[str]:
    """Collect all sentences from a document row.

    Extracts sentences from title, abstract, and fulltext fields.

    Args:
        row: DataFrame row (namedtuple from itertuples).
        nlp_abstract: Processed abstract spaCy Doc (or None).
        nlp_fulltext: Processed fulltext spaCy Doc (or None).
        columns: Set of column names in the DataFrame.

    Returns:
        List of sentence strings.
    """
    sent_batch = []

    if "title" in columns:
        sent_batch.append(row["title"].strip())

    if "abstract" in columns and nlp_abstract is not None:
        sent_batch.extend(s.text for s in nlp_abstract.sents if s.text)

    if "fulltext" in columns and nlp_fulltext is not None:
        sent_batch.extend(s.text for s in nlp_fulltext.sents if s.text)

    return sent_batch


def process_document(
    doc_id: str,
    doc_row: int,
    sentences: list[str],
    keywords: list[str],
    tokenizer: AutoTokenizer,
    stemmer: PorterStemmer,
    scheme: str = "IOB",
    device: str = "cpu",
    match_mode: Literal["stem_equality", "stem_regex"] = "stem_equality",
    truncation: bool = True,
    padding: bool = True,
) -> list[dict]:
    """Process a single document through the full pipeline.

    Tokenizes sentences, matches keywords, and creates training examples.

    Args:
        doc_id: Source file identifier.
        doc_row: Row number in source file.
        sentences: List of sentence strings.
        keywords: List of document keywords.
        tokenizer: HuggingFace tokenizer.
        stemmer: Porter stemmer.
        scheme: Labeling scheme - "IOB" (0=Outside, 1=Beginning, 2=Inside) or
            "binary" (0=not keyword, 1=keyword). Defaults to "IOB".
        device: Torch device.
        match_mode: Keyword matching strategy.
            - "stem_equality" (default): word-level stem equality matching,
              equivalent to the legacy gold-standard approach. Correctly handles
              edge cases like keyword "ai" not matching "aims".
            - "stem_regex": exact word-boundary regex matching via
              build_keyword_regex / resolve_keyword_spans.
        truncation: Whether to truncate sequences exceeding model_max_length.
        padding: Whether to pad sequences to longest in batch.

    Returns:
        List of processed sentence dictionaries ready for Parquet output.
    """
    processed = []

    # Normalize keywords to match legacy behavior (lowercase + replace -, /, ∗ with spaces)
    normalized_keywords = [_normalize_text(kw).lower() for kw in keywords]

    # Tokenize all sentences at once (on original text for HF tokenizer)
    tokenized = tokenize_sentences(
        sentences, tokenizer, truncation=truncation, padding=padding
    )

    # Extract word_ids and word_lists using tokenizer word boundaries
    word_ids_per_sent, word_lists_per_sent = extract_word_level_data(
        tokenized, sentences
    )

    offset_mapping = tokenized.pop("offset_mapping")

    # Build regex pattern once (only needed for stem_regex mode)
    regex_pattern = (
        build_keyword_regex(normalized_keywords, stemmer)
        if match_mode == "stem_regex"
        else None
    )

    for s_idx, sent in enumerate(sentences):
        sent_tok_len = tokenized.attention_mask[s_idx].sum().item()

        # Normalize sentence text for keyword matching (positions preserved since
        # -, /, ∗ are all single-char → single-space replacements)
        sent_normalized = _normalize_text(sent).lower()

        # Resolve keywords in this sentence
        if match_mode == "stem_equality":
            sent_kwds, tok_ranges = resolve_keyword_spans_stem(
                sent_normalized,
                offset_mapping[s_idx].numpy(),
                sent_tok_len,
                normalized_keywords,
                stemmer,
            )
        else:
            sent_kwds, tok_ranges = resolve_keyword_spans(
                sent_normalized,
                offset_mapping[s_idx].numpy(),
                sent_tok_len,
                regex_pattern,
            )

        # Create keyword mask
        keyword_mask = create_keyword_mask(
            tok_ranges,
            sent_tok_len,
            scheme,
            device,
            word_ids=word_ids_per_sent[s_idx][:sent_tok_len],
        )

        processed.append(
            {
                "doc_id": doc_id,
                "doc_row": doc_row,
                "sent_num": s_idx,
                "sent_txt": sent,
                "keywords": keywords,
                "sent_keywords": sent_kwds,
                "token_ids": tokenized.input_ids[s_idx, :sent_tok_len].tolist(),
                "attention_mask": tokenized.attention_mask[
                    s_idx, :sent_tok_len
                ].tolist(),
                "keyword_mask": keyword_mask.tolist(),
                "word_ids": word_ids_per_sent[s_idx][:sent_tok_len],
                "word_list": word_lists_per_sent[s_idx],
            }
        )

    return processed


def process_dataset(
    filepath: Path,
    nlp: spacy.Language,
    spacy_batch_size: int,
    spacy_n_process: int,
    tokenizer: AutoTokenizer,
    stemmer: PorterStemmer,
    scheme: str = "IOB",
    limit: int | None = None,
    device: str = "cpu",
    show_progress: bool = True,
    match_mode: Literal["stem_equality", "stem_regex"] = "stem_equality",
    truncation: bool = True,
    padding: bool = True,
) -> list[dict]:
    """Process a single JSON file through the full pipeline.

    Reads a JSON lines file, segments sentences, tokenizes, matches
    keywords, and returns processed data ready for Parquet output.

    Args:
        filepath: Path to JSON file.
        nlp: Loaded spaCy model.
        spacy_batch_size: Batch size for spaCy sentence segmentation.
        spacy_n_process: Number of processes for spaCy parallel processing.
        tokenizer: HuggingFace tokenizer.
        stemmer: Porter stemmer.
        scheme: Labeling scheme - "IOB" (0=Outside, 1=Beginning, 2=Inside) or
            "binary" (0=not keyword, 1=keyword). Defaults to "IOB".
        limit: Optional row limit for testing.
        device: Torch device.
        show_progress: Whether to show progress bar.
        match_mode: Keyword matching strategy, passed through to process_document.
            See process_document for details.
        truncation: Whether to truncate sequences exceeding model_max_length.
        padding: Whether to pad sequences to longest in batch.

    Returns:
        List of processed sentence dictionaries.
    """
    # Read JSON file
    # df = pd.read_json(filepath, lines=True)
    df = pl.read_ndjson(filepath)
    if limit is not None:
        df = df.head(limit)

    columns = frozenset(df.columns)
    doc_id = str(filepath)

    # Process abstracts and fulltexts with spaCy (batch processing)
    nlp_abstracts: list[spacy.tokens.Doc | None]
    nlp_fulltexts: list[spacy.tokens.Doc | None]

    if "abstract" in columns:
        logger.info(
            f"Segmenting abstracts (batch_size: {spacy_batch_size}, n_process: {spacy_n_process})"
        )
        nlp_abstracts = segment_sentences(
            df["abstract"].to_list(),
            nlp,
            batch_size=spacy_batch_size,
            n_process=spacy_n_process,
        )
    else:
        nlp_abstracts = [None] * len(df)

    if "fulltext" in columns:
        logger.info(
            f"Segmenting full texts (batch_size: {spacy_batch_size}, n_process: {spacy_n_process})"
        )
        nlp_fulltexts = segment_sentences(
            df["fulltext"].to_list(),
            nlp,
            batch_size=spacy_batch_size,
            n_process=spacy_n_process,
        )
    else:
        nlp_fulltexts = [None] * len(df)

    # Process each document
    all_processed = []
    iterator = zip(
        df.with_row_index().iter_rows(named=True), nlp_abstracts, nlp_fulltexts
    )

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(df),
            desc=filepath.name,
            leave=False,
        )

    for i, (row, nlp_abs, nlp_fulltxt) in enumerate(iterator):
        # Extract keywords
        keywords = extract_keywords(row, columns)
        if not keywords:
            continue

        # Collect sentences from document
        sentences = collect_sentences(row, nlp_abs, nlp_fulltxt, columns)
        if not sentences:
            continue

        # Process document
        doc_processed = process_document(
            doc_id=doc_id,
            doc_row=i,
            sentences=sentences,
            keywords=keywords,
            tokenizer=tokenizer,
            stemmer=stemmer,
            scheme=scheme,
            device=device,
            match_mode=match_mode,
            truncation=truncation,
            padding=padding,
        )
        all_processed.extend(doc_processed)

    return all_processed
