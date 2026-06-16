"""
logging_utils.py
================
Centralised logging setup for the ISRO AQI & HCHO project.

Usage::

    from src.utils.logging_utils import get_logger, setup_logging

    # Module-level logger (standard pattern)
    logger = get_logger(__name__)

    # Optional: call once at entry-point to add a rotating file handler
    setup_logging(log_file="logs/pipeline.log", level="INFO")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """
    Return a named logger with a StreamHandler pointing to stdout.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : str | int
        Logging level, e.g. "INFO", "DEBUG", logging.WARNING.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def setup_logging(
    log_file: str | Path | None = None,
    level: str | int = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    Configure the root logger with an optional rotating file handler.

    Call once at the entry point of a script (e.g. train_aqi.py ``main()``).

    Parameters
    ----------
    log_file : str | Path | None
        If provided, a RotatingFileHandler is added to the root logger.
    level : str | int
        Root logging level.
    max_bytes : int
        Maximum bytes per log file before rotation (default 5 MB).
    backup_count : int
        Number of backup files to keep.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        root.addHandler(fh)
        logging.getLogger(__name__).info("Log file: %s", log_path.resolve())
