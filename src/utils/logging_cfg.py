# src/utils/logging_cfg.py
"""
Centralized logging configuration for the decision-tree-privacy-scan project.

Provides a factory function to obtain a named logger that writes INFO-level
logs to the console and DEBUG-level logs to a rotating file in the `logs/` directory.

Usage:
    from utils.logging_cfg import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str = "dt_privacy_scan") -> logging.Logger:
    """
    Return a configured logger with both console and rotating file handlers.

    - Console (stdout): level INFO
    - File (logs/<name>.log): level DEBUG, rotates at 2 MB, keeps 3 backups

    Subsequent calls with the same name reuse existing handlers.

    Parameters
    ----------
    name : str
        The logger name (typically __name__ in client modules).

    Returns
    -------
    logging.Logger
        A ready-to-use logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(FMT)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_DIR / f"{name}.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FMT)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
