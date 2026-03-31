"""Centralised logging configuration for docswarm."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/docswarm.log and to stdout.

    The log directory is created automatically.  Calling this function
    multiple times with the same *name* returns the same logger instance
    (standard Python logging behaviour).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "docswarm.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
