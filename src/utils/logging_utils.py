"""
logging_utils.py
================
Logging setup shared across training and evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def create_logger(
    log_dir: Optional[str] = None,
    name: str = __name__,
    log_file: str = "train.log",
) -> logging.Logger:
    """Create a logger that writes to both console and an optional file.

    Args:
        log_dir: Directory for the log file. ``None`` → console only.
        name:    Logger name (defaults to this module's ``__name__``).
        log_file: Filename inside *log_dir*.

    Returns:
        Configured :class:`logging.Logger`.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(
            logging.FileHandler(os.path.join(log_dir, log_file))
        )

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(name)
