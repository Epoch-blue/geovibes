#!/usr/bin/env python3
"""
Filter palm oil mills GeoJSON to remove entries from a specific year.

Usage:
    uv run scripts/filter_mills_by_year.py
"""

import json
from pathlib import Path


def filter_mills(
    input_path: str,
    output_path: str,
    exclude_year: str = "2025",
) -> None:
    """Remove mills with earliest_year_of_existence matching exclude_year."""
    with open(input_path) as f:
        data = json.load(f)

    original_count = len(data["features"])

    filtered_features = [
        f for f in data["features"]
        if f.get("properties", {}).get("earliest_year_of_existence") != exclude_year
    ]

    removed_count = original_count - len(filtered_features)

    data["features"] = filtered_features

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Original:  {original_count} mills")
    print(f"Removed:   {removed_count} mills (year={exclude_year})")
    print(f"Remaining: {len(filtered_features)} mills")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    input_file = Path("geometries/indonesia-palm-oil-mills.geojson")
    output_file = Path("geometries/indonesia-palm-oil-mills-filtered.geojson")

    filter_mills(str(input_file), str(output_file), exclude_year="2025")
