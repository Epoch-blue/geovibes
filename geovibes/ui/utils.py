"""Utility helpers for the GeoVibes UI package."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

LOG_FILE = "geovibes_crash.log"

def prepare_ids_for_query(ids: Iterable[object]) -> List[str]:
    """Normalize identifiers before passing them to DuckDB."""
    return [str(identifier) for identifier in ids]


def log_to_file(message: str, logfile: str = LOG_FILE) -> None:
    """Append a timestamped message to the crash log."""
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} - {message}\n")


__all__ = ["prepare_ids_for_query", "log_to_file", "LOG_FILE"]
