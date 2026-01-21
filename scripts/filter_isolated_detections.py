#!/usr/bin/env python3
"""
Filter out isolated detections that don't touch any other detections.

Real features like palm oil mills typically produce clusters of adjacent
detections. Isolated single-tile detections are often noise.

Usage:
    uv run scripts/filter_isolated_detections.py \
        --input classification_output/classification_detections.geojson \
        --output classification_output/classification_detections_clustered.geojson

    # Require at least 3 touching detections
    uv run scripts/filter_isolated_detections.py \
        --input classification_output/classification_detections.geojson \
        --output classification_output/classification_detections_clustered.geojson \
        --min-neighbors 2
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely import STRtree


def filter_isolated(
    input_path: str,
    output_path: str,
    min_neighbors: int = 1,
) -> dict:
    """
    Remove detections that don't touch at least min_neighbors other detections.

    Parameters
    ----------
    input_path : str
        Path to input detections GeoJSON
    output_path : str
        Path to write filtered detections
    min_neighbors : int
        Minimum number of touching neighbors required (default: 1)

    Returns
    -------
    dict
        Statistics about the filtering
    """
    print(f"Loading {input_path}...")
    gdf = gpd.read_file(input_path)
    original_count = len(gdf)
    print(f"Loaded {original_count:,} detections")

    print(
        f"Building spatial index and finding neighbors (min_neighbors={min_neighbors})..."
    )

    # Build spatial index for fast neighbor queries
    tree = STRtree(gdf.geometry)

    # For each geometry, count how many others it touches
    neighbor_counts = []
    for idx, geom in enumerate(gdf.geometry):
        # Query geometries that might intersect (using bounding box)
        candidate_indices = tree.query(geom)

        # Count actual touches (excluding self)
        touch_count = 0
        for candidate_idx in candidate_indices:
            if candidate_idx != idx:
                if geom.touches(gdf.geometry.iloc[candidate_idx]) or geom.intersects(
                    gdf.geometry.iloc[candidate_idx]
                ):
                    touch_count += 1

        neighbor_counts.append(touch_count)

    gdf["neighbor_count"] = neighbor_counts

    # Filter to keep only detections with enough neighbors
    filtered_gdf = gdf[gdf["neighbor_count"] >= min_neighbors].copy()

    # Drop the neighbor_count column before saving
    filtered_gdf = filtered_gdf.drop(columns=["neighbor_count"])

    filtered_count = len(filtered_gdf)
    removed_count = original_count - filtered_count

    print(f"Saving {filtered_count:,} detections to {output_path}...")
    filtered_gdf.to_file(output_path, driver="GeoJSON")

    # Compute statistics
    isolated_count = sum(1 for n in neighbor_counts if n == 0)
    stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": removed_count,
        "isolated_count": isolated_count,
        "min_neighbors": min_neighbors,
    }

    print(f"\n{'=' * 60}")
    print("FILTER RESULTS")
    print(f"{'=' * 60}")
    print(f"Original detections:   {original_count:,}")
    print(
        f"Isolated (0 neighbors): {isolated_count:,} ({isolated_count / original_count * 100:.1f}%)"
    )
    print(
        f"Removed (< {min_neighbors} neighbors): {removed_count:,} ({removed_count / original_count * 100:.1f}%)"
    )
    print(
        f"Remaining:             {filtered_count:,} ({filtered_count / original_count * 100:.1f}%)"
    )
    print(f"{'=' * 60}")

    # Show neighbor distribution
    print("\nNeighbor count distribution:")
    for n in range(min(10, max(neighbor_counts) + 1)):
        count = sum(1 for nc in neighbor_counts if nc == n)
        pct = count / original_count * 100
        bar = "#" * int(pct / 2)
        print(f"  {n} neighbors: {count:>6,} ({pct:>5.1f}%) {bar}")

    if max(neighbor_counts) >= 10:
        count = sum(1 for nc in neighbor_counts if nc >= 10)
        pct = count / original_count * 100
        print(f"  10+ neighbors: {count:>5,} ({pct:>5.1f}%)")

    print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter out isolated detections that don't touch others"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input detections GeoJSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output filtered GeoJSON",
    )
    parser.add_argument(
        "--min-neighbors",
        "-n",
        type=int,
        default=1,
        help="Minimum number of touching neighbors required (default: 1)",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    filter_isolated(
        input_path=args.input,
        output_path=args.output,
        min_neighbors=args.min_neighbors,
    )


if __name__ == "__main__":
    main()
