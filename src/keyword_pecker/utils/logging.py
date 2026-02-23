"""Minimal logging utilities."""

import logging
import sys


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        return f"[{level}] {record.name} | {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stderr with the project format.

    Output format: ``[LEVEL] name | message``

    Args:
        name: Logger name, typically the module or script name.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_Formatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger
