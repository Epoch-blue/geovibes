"""Utility helpers for the GeoVibes UI package."""

from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

LOG_FILE = "geovibes_crash.log"


def prepare_ids_for_query(ids: Iterable[object]) -> List[str]:
    return [str(identifier) for identifier in ids]


def log_to_file(message: str, logfile: str = LOG_FILE) -> None:
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} - {message}\n")


def list_databases_in_directory(
    directory_path: str, verbose: bool = False
) -> List[Dict[str, str]]:
    databases: List[Dict[str, str]] = []
    pattern = os.path.join(directory_path, "*.db")
    for db_file in glob.glob(pattern):
        if not os.path.isfile(db_file):
            continue
        base_name, _ = os.path.splitext(db_file)
        if base_name.endswith("_metadata"):
            prefix = base_name[: -len("_metadata")]
            index_pattern = f"{prefix}*.index"
        else:
            index_pattern = f"{base_name}*.index"
        index_files = glob.glob(index_pattern)
        if len(index_files) == 1:
            databases.append({"db_path": db_file, "faiss_path": index_files[0]})
            if verbose:
                print(f"  Found DB: {db_file} with Index: {index_files[0]}")
        elif verbose:
            if index_files:
                print(f"⚠️  Multiple index files found for {db_file}: {index_files}")
            else:
                print(f"  Found DB: {db_file}, but no associated FAISS index found.")
    if verbose:
        print(f"Found {len(databases)} database(s) in {directory_path}")

    # google is most lightweight, start here
    def sort_key(entry: Dict[str, str]) -> tuple[int, str]:
        db_name = Path(entry['db_path']).stem.lower()
        if 'alabama' in db_name and 'google' in db_name:
            return (0, entry['db_path'])
        if db_name.startswith('alabama_earthgenome_softcon'):
            return (1, entry['db_path'])
        return (2, entry['db_path'])

    return sorted(databases, key=sort_key)


_TILE_SPEC_PATTERN = re.compile(
    r"(?P<size>\d+?)_(?P<overlap>\d+?)_(?P<resolution>\d+(?:\.\d+)?)$"
)


def infer_tile_spec_from_name(name: str) -> Optional[Dict[str, float]]:
    base = Path(name).stem if "." in name else name
    if base.endswith("_metadata"):
        base = base[: -len("_metadata")]
    match = _TILE_SPEC_PATTERN.search(base)
    if not match:
        return None
    size = int(match.group("size"))
    overlap = int(match.group("overlap"))
    resolution = float(match.group("resolution"))
    if size <= 0 or resolution <= 0:
        return None
    return {
        "tile_size_px": size,
        "tile_overlap_px": overlap,
        "meters_per_pixel": resolution,
    }


def get_database_centroid(duckdb_connection, verbose: bool = False) -> tuple[float, float]:
    if verbose:
        print("📍 Using default center (0, 0)")
    return 0.0, 0.0


__all__ = [
    "prepare_ids_for_query",
    "log_to_file",
    "LOG_FILE",
    "list_databases_in_directory",
    "infer_tile_spec_from_name",
    "get_database_centroid",
]
