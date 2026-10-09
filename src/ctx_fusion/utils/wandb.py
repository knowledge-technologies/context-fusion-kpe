"""Optional Weights & Biases integration.

Provides a single setup function that performs the one-time availability check
and initialises a W&B run.  The returned module reference (or ``None``) is then
used directly by callers — no wrapper functions needed::

    import wandb as wb

    wb, run_name = setup_wandb(enabled=args.wandb, config=hyperparams, ...)

    # later in the training loop:
    if wb:
        wb.log({"train/loss": loss}, step=step)

    # at the end of training:
    if wb:
        wb.finish()
"""

import os
import types
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

from ctx_fusion.utils.logging import get_logger

logger = get_logger(__name__)

load_dotenv()


def _timestamp_name() -> str:
    """Return a timestamp-based fallback run name, e.g. ``run_20260223_143012``."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def setup_wandb(
    *,
    enabled: bool,
    config: dict[str, Any],
    project: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> tuple[types.ModuleType | None, str]:
    """Initialise W&B and return ``(wandb_module, run_name)``.

    The import check and ``wandb.init()`` call happen exactly once here.
    If W&B is disabled or not installed, ``None`` is returned as the module so
    callers can gate all subsequent W&B calls with a simple ``if wandb:``.

    Args:
        enabled: Whether to attempt W&B initialisation.
        config: Hyperparameter dict logged to ``wandb.config``.
        project: W&B project name.  Falls back to ``WANDB_PROJECT`` env var,
            then ``"ctx-fusion"``.
        entity: W&B entity (team/user).  Falls back to ``WANDB_ENTITY`` env var.
        tags: Optional list of tags attached to the run.
        notes: Free-text notes stored with the run.

    Returns:
        A ``(wandb_module_or_None, run_name)`` tuple.  ``run_name`` is the W&B
        adjective-noun name (e.g. ``"misty-river-42"``) when W&B is active, or
        a timestamp string (e.g. ``"run_20260223_143012"``) otherwise.
    """
    if not enabled:
        return None, _timestamp_name()

    try:
        import wandb
    except ImportError:
        logger.warning(
            "wandb is not installed — W&B tracking disabled. "
            "Install it with: uv pip install 'ctx-fusion[tracking]'"
        )
        return None, _timestamp_name()

    resolved_project = project or os.environ.get("WANDB_PROJECT", "ctx-fusion")
    resolved_entity = entity or os.environ.get("WANDB_ENTITY") or None

    run = wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        config=config,
        tags=tags or [],
        notes=notes,
    )

    run_name: str = run.name  # type: ignore[union-attr]
    logger.info(f"W&B run initialised: {run_name} (project={resolved_project})")
    return wandb, run_name
