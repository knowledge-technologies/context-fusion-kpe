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
from typing import Any

import numpy as np
import polars as pl
import regex
import spacy
import torch
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from transformers import AutoTokenizer

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


def build_keyword_regex(
    keywords: list[str],
    stemmer: PorterStemmer,
) -> tuple[regex.Pattern, list[str]]:
    """Build stemmed regex pattern from keywords.

    Creates a compiled regex that matches stemmed versions of keywords,
    sorted by length (longest first) to prioritize longer matches.

    Args:
        keywords: List of keyword phrases.
        stemmer: Porter stemmer instance.

    Returns:
        Tuple of (compiled_regex_pattern, stemmed_keywords).

    Example:
        >>> stemmer = PorterStemmer()
        >>> pattern, stemmed = build_keyword_regex(["machine learning"], stemmer)
        >>> pattern.pattern
        'machin\\\\w*\\\\s+learn\\\\w*'
    """
    regex_parts = []
    stemmed_kws = []

    for kw in keywords:
        word_re_parts = []
        stemmed_parts = []
        for w in kw.split():
            stemmed = stemmer.stem(w)
            stemmed_parts.append(stemmed)
            word_re_parts.append(rf"{regex.escape(stemmed)}\w*")
        stemmed_kws.append(" ".join(stemmed_parts))
        regex_parts.append(r"\s+".join(word_re_parts))

    # Sort by length (longest first) to prioritize longer matches
    combined_pattern = "|".join(sorted(regex_parts, key=len, reverse=True))
    return regex.compile(combined_pattern, flags=regex.IGNORECASE), stemmed_kws


def resolve_keyword_spans(
    sentence: str,
    offsets: np.ndarray,
    num_total_tokens: int,
    regex_pattern: regex.Pattern,
    stemmed_kws: list[str],
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
        stemmed_kws: List of stemmed keywords (for reference).

    Returns:
        Tuple of (matched_keywords, token_ranges):
        - matched_keywords: List of keyword strings found in sentence.
        - token_ranges: List of (start_tok, end_tok) tuples.

    Note:
        Handles overlapping keywords by skipping subsequent matches.
        Handles truncation by stopping when keyword extends beyond tokens.
    """
    kwds, word_token_slice_ixs = [], []

    # token_boundaries is an array with 1 on places where a new token
    # begins. we will take a cumulative sum such that all array
    # positions map to the index of the corresponding token. we ignore
    # first/last offset because it relates to <BOS>/<EOS> tokens and
    # offset does not map to it (ie. both start and end are 0; and
    # first word again starts with zero). initial array starts with 1,
    # and that is fine as word vectors also start from index 1 (after
    # <BOS>)
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
            # this is an unfortunate case where keywords share
            # some words. for this case we first find the longest
            # match
            #
            # TODO - overlapping keywords (text contains 'medical
            # knowledge flow barriers', while keywords are
            # 'medical knowledge' and 'knowledge flow barriers')
            #
            # - we can generate contextual embeddings and do a
            # meshgrid of those embeddings (meshgrid(embs,
            # embs)). each item of this grid is categorized as a
            # span containing the keyword or not. most of the word
            # to adapt it is in preprocessing. we can store spans
            # (start_tok_idx:int, end_tok_idx:int) to db and only
            # construct the target array after the inference
            #
            continue

        if w_end > last_token_ends_at:
            # if word ends after this point, the sentence was
            # truncated and we can only skip the rest
            break

        # word is in slice [w_start, w_end>
        kwds.append(w_string)

        # we subtract 1 from w_end as this w_end is a slice (w_start,
        # w_end), real position ends at w_end-1.
        #
        # we add one to final token_boundary because we are again
        # creating a slice, and slice of size 1 should start at i and
        # end on i+1. without it it would be (i, i)
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


# ============================================================================
# Tokenization & Keyword Alignment
# ============================================================================


def tokenize_sentences(
    sentences: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """Tokenize sentences with offset mapping.

    Args:
        sentences: List of sentence texts.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Tokenizer output dict with input_ids, attention_mask, offset_mapping.
    """
    return tokenizer(
        sentences,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
    )


def create_keyword_mask(
    keyword_token_ranges: list[tuple[int, int]],
    sent_tok_len: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Create binary mask for tokens that are part of keywords.

    Args:
        keyword_token_ranges: List of (start, end) token index pairs.
        sent_tok_len: Total number of tokens in sentence.
        device: Torch device for computation.

    Returns:
        Binary tensor of shape (sent_tok_len,) with 1 for keyword tokens.

    Raises:
        ValueError: If overlapping keywords detected (should not happen
            with the skip strategy in resolve_keyword_spans).
    """
    if not keyword_token_ranges:
        return torch.zeros(sent_tok_len, dtype=torch.long)

    tok_start, tok_end = zip(*keyword_token_ranges)
    tok_start = torch.tensor(tok_start)
    tok_end = torch.tensor(tok_end)
    arange = torch.arange(sent_tok_len).tile(len(keyword_token_ranges), 1)
    targets = torch.ones_like(arange, device=device)
    targets[arange < tok_start.view(-1, 1)] = 0
    targets[arange >= tok_end.view(-1, 1)] = 0
    targets = targets.sum(dim=0)

    # Check for overlapping keywords (should not happen with skip strategy)
    if targets.max() > 1:
        raise ValueError(
            f"Overlapping keywords detected: max overlap = {targets.max()}"
        )

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

    df = pl.DataFrame(data_rows, schema=schema or SENTENCE_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="snappy")
    return len(df)


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
    device: str = "cpu",
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
        device: Torch device.

    Returns:
        List of processed sentence dictionaries ready for Parquet output.
    """
    processed = []

    # Build keyword regex pattern
    regex_pattern, stemmed_kws = build_keyword_regex(keywords, stemmer)

    # Tokenize all sentences at once
    tokenized = tokenize_sentences(sentences, tokenizer)
    offset_mapping = tokenized.pop("offset_mapping")

    for s_idx, sent in enumerate(sentences):
        sent_tok_len = tokenized.attention_mask[s_idx].sum().item()

        # Resolve keywords in this sentence
        sent_kwds, tok_ranges = resolve_keyword_spans(
            sent.lower(),
            offset_mapping[s_idx].numpy(),
            sent_tok_len,
            regex_pattern,
            stemmed_kws,
        )

        # Create keyword mask
        keyword_mask = create_keyword_mask(tok_ranges, sent_tok_len, device)

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
    limit: int | None = None,
    device: str = "cpu",
    show_progress: bool = True,
) -> list[dict]:
    """Process a single JSON file through the full pipeline.

    Reads a JSON lines file, segments sentences, tokenizes, matches
    keywords, and returns processed data ready for Parquet output.

    Args:
        filepath: Path to JSON file.
        nlp: Loaded spaCy model.
        tokenizer: HuggingFace tokenizer.
        stemmer: Porter stemmer.
        limit: Optional row limit for testing.
        device: Torch device.
        show_progress: Whether to show progress bar.

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
            device=device,
        )
        all_processed.extend(doc_processed)

    return all_processed
