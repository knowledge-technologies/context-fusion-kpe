"""Run directory utilities for experiment management.

This module provides utilities for managing experiment run directories,
including creating and locating run folders for storing checkpoints,
metrics, and other training artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_runs_root() -> Path:
    """Return the root directory for all experiment runs.

    Reads ``EXPERIMENT_RUNS_FOLDER`` from the environment (set in ``.env``).
    Falls back to ``$DATASET_FOLDER/runs`` if not defined.
    Last resort: write next to the project root.

    Returns:
        Path to the runs root directory.
    """
    runs_env = os.environ.get("EXPERIMENT_RUNS_FOLDER")
    if runs_env:
        return Path(runs_env)

    dataset_env = os.environ.get("DATASET_FOLDER")
    if dataset_env:
        return Path(dataset_env) / "runs"

    return Path(__file__).parent.parent.parent.parent / "runs"


def init_run_dir(run_name: str) -> Path:
    """Create and return the directory for a single run.

    Args:
        run_name: Name of the run (used as directory name).

    Returns:
        Path to the created run directory.
    """
    run_dir = get_runs_root() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
