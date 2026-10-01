"""Bundled external data for manuscript reproduction."""

from pathlib import Path

EXTERNAL_DIR = Path(__file__).parent

BENCHMARKS_EVALS_PATH = EXTERNAL_DIR / "benchmarks_evals.json"
SEKE_TRAINSET_PATH = EXTERNAL_DIR / "seke_evals_trainset.json"
SEKE_VALSET_PATH = EXTERNAL_DIR / "seke_evals_valset.json"
