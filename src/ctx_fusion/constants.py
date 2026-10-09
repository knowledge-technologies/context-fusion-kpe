"""Project-wide constants."""

# PyTorch CrossEntropyLoss ignore index — labels set to this value are excluded from loss computation.
IGNORE_IDX = -100

# Punctuation characters to disallow in keyphrases.

# Punctuation characters to disallow in keyphrases.
# Copied exactly from the legacy TNT-KID eval code (train_and_eval.py line 858).
# NOTE: This includes apostrophe ('), which contradicts the original TNT-KID paper
# stating "we remove keyphrases containing punctuation (except dashes and apostrophes)".
# We keep the legacy behavior as-is for reproducible evaluation comparisons.
DEFAULT_DISALLOWED_PUNCTUATION = "!#$%&'()*+,.:;<=>?@[\\]^_`{|}~"
