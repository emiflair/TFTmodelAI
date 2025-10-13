"""Logging utilities for consistent project logging."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import ARTIFACT_ROOT


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a configured logger that logs to stdout and optional file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = ARTIFACT_ROOT / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_path) / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
