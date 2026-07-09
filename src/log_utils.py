"""Adds a per-run file handler on top of the console logging each script
already sets up via `logging.basicConfig`, so training runs leave a
persistent record under `logs/` instead of only stdout."""
from __future__ import annotations

import logging
from pathlib import Path


def add_file_handler(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
