#!/usr/bin/env python3
"""
region_to_embeddings.py
-----------------------
Download Earth Genome *earthindexembeddings* Parquet files that spatially
intersect a user‑supplied region **in parallel**.

Changes vs. v1
~~~~~~~~~~~~~~
* **Parallel downloads** with a configurable `--workers` argument (default: 8) via
  `concurrent.futures.ThreadPoolExecutor`.
* Minor refactor: the S3 client is instantiated *per worker* to avoid potential
  thread‑safety issues.
* Progress information for each completed download.

Overview
~~~~~~~~
Given a GeoJSON **or** GeoParquet file describing one or more polygons, the
script:

1. Converts the region(s) to WGS84 (EPSG:4326).
2. Samples a regular lon/lat grid over the bounding box (default step
   0.05° ≈ 5 – 6 km).
3. Maps each grid point to a 100 km **MGRS** tile key (first 5 chars).
4. Constructs S3 object paths of the form
     `s3://earthgenome/earthindexembeddings/2024/<TILE>_<START>_<END>.parquet`.
5. Downloads each Parquet file concurrently to a local directory, using the
   custom endpoint `https://data.source.coop` with unsigned requests.
6. Logs missing keys and a summary at the end.

Dependencies
~~~~~~~~~~~~
```
pip install geopandas shapely numpy mgrs boto3 botocore pyarrow
```

Example
~~~~~~~
```bash
python region_to_embeddings.py region.gpq \
    --start 2024-01-01 --end 2025-01-01 \
    --out-dir ./embeddings --workers 16 --verbose
```
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import logging
import os
import pathlib
import sys
from typing import Iterable, Set

import boto3
import botocore
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

_LOG = logging.getLogger("region_to_embeddings")

S3_BUCKET = "earthgenome"
S3_PREFIX = "earthindexembeddings/2024/"
DEFAULT_ENDPOINT = "https://data.source.coop"
DEFAULT_MGRS_TILE_ID_COLUMN = "tile"
DEFAULT_WORKERS = 8

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Earth Genome embedding parquet files intersecting a region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_file", help="GeoJSON or GeoParquet file defining region(s) of interest")
    p.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    p.add_argument("--out-dir", default="embeddings", help="Local directory for downloads")
    p.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT, help="S3-compatible endpoint URL")
    p.add_argument("--mgrs-reference-file", required=True, help="Geospatial file (e.g., GeoJSON, GeoParquet) containing MGRS tile geometries and IDs")
    p.add_argument("--mgrs-tile-id-column", default=DEFAULT_MGRS_TILE_ID_COLUMN, help="Column name in mgrs-reference-file that contains the MGRS tile identifier")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel download threads")
    p.add_argument("--dry-run", action="store_true", help="List files but do not download")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _load_region(path: str | pathlib.Path) -> gpd.GeoSeries:
    """Load a vector file (GeoJSON or GeoParquet) and return one unified geometry in WGS84."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".gpq", ".parquet"}:
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError("No geometries found in input file")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")


def _get_intersecting_mgrs_ids_from_reference(
    region_gs: gpd.GeoSeries, 
    mgrs_reference_filepath: str | pathlib.Path, 
    mgrs_tile_id_column: str
) -> Set[str]:
    """
    Identifies MGRS tile IDs by intersecting the input region with a reference MGRS tile file.
    """
    _LOG.info(f"Loading MGRS reference file from {mgrs_reference_filepath}")
    mgrs_reference_filepath = pathlib.Path(mgrs_reference_filepath)
    if not mgrs_reference_filepath.exists():
        raise FileNotFoundError(f"MGRS reference file not found: {mgrs_reference_filepath}")

    if mgrs_reference_filepath.suffix.lower() in {".gpq", ".parquet"}:
        mgrs_gdf = gpd.read_parquet(mgrs_reference_filepath)
    else:
        mgrs_gdf = gpd.read_file(mgrs_reference_filepath)

    if mgrs_gdf.empty:
        _LOG.warning("MGRS reference file is empty.")
        return set()

    if mgrs_tile_id_column not in mgrs_gdf.columns:
        raise ValueError(
            f"MGRS tile ID column '{mgrs_tile_id_column}' not found in MGRS reference file. "
            f"Available columns: {mgrs_gdf.columns.tolist()}"
        )

    # Ensure MGRS reference GDF is in EPSG:4326 if CRS is defined
    if mgrs_gdf.crs is not None and mgrs_gdf.crs.to_epsg() != 4326:
        _LOG.info(f"Reprojecting MGRS reference layer from {mgrs_gdf.crs} to EPSG:4326.")
        mgrs_gdf = mgrs_gdf.to_crs(4326)
    elif mgrs_gdf.crs is None:
        _LOG.warning("MGRS reference file has no CRS defined. Assuming EPSG:4326.")
        mgrs_gdf.crs = "EPSG:4326" # Assume WGS84 if not set; risky but common for GeoJSONs

    # Prepare the input region for spatial join (ensure it's a GeoDataFrame)
    # The input region_gs is already in EPSG:4326 from _load_region
    region_gdf = gpd.GeoDataFrame(geometry=region_gs)

    _LOG.info("Performing spatial join between input region and MGRS reference layer...")
    # Perform spatial join. 'inner' keeps only intersecting features.
    # predicate='intersects' is default for sjoin if not specified
    intersecting_mgrs_tiles = gpd.sjoin(mgrs_gdf, region_gdf, how="inner", predicate="intersects")

    if intersecting_mgrs_tiles.empty:
        _LOG.info("No intersecting MGRS tiles found.")
        return set()

    tile_keys = set(intersecting_mgrs_tiles[mgrs_tile_id_column].unique())
    _LOG.info(f"Found {len(tile_keys)} unique intersecting MGRS tile IDs.")
    return tile_keys

# -----------------------------------------------------------------------------
# S3 helpers
# -----------------------------------------------------------------------------

def _new_s3_client(endpoint_url: str):
    """Create a new unsigned boto3 S3 client (safe for thread local use)."""
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )


def _download_single(key_info: tuple[str, pathlib.Path, str, bool]):
    """Worker: download one S3 key -> local filepath.

    Parameters
    ----------
    key_info : tuple
        (s3_key, dst_path, endpoint_url, dry_run)
    """
    s3_key, dst_path, endpoint_url, dry_run = key_info
    client = _new_s3_client(endpoint_url)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return f"[dry-run] {s3_key} -> {dst_path}"

    try:
        client.download_file(S3_BUCKET, s3_key, str(dst_path))
        return f"Downloaded {s3_key} -> {dst_path}"  # success
    except client.exceptions.NoSuchKey:
        return f"Missing {s3_key} (not found)"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    args = _parse_args(argv or sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(levelname)s: %(message)s",
    )

    # Validate dates
    logging.info(f"Start date: {args.start}")
    logging.info(f"End date: {args.end}")
    try:
        start_date = _dt.date.fromisoformat(args.start)
        end_date = _dt.date.fromisoformat(args.end)
    except ValueError as e:
        _LOG.error("Invalid date: %s", e)
        sys.exit(1)

    # Load region & compute tiles
    logging.info(f"Loading region from {args.input_file}")
    region = _load_region(args.input_file)
    logging.info(f"Region loaded: {region}")
    
    # Use the new function to get MGRS tiles
    tiles = _get_intersecting_mgrs_ids_from_reference(
        region, 
        args.mgrs_reference_file, 
        args.mgrs_tile_id_column
    )
    logging.info(f"Tiles: {tiles}")
    _LOG.info("Identified %d tile(s).", len(tiles))

    # Build S3 keys & local paths
    logging.info(f"Building S3 keys")
    s3_keys = [f"{S3_PREFIX}{tile}_{start_date}_{end_date}.parquet" for tile in sorted(tiles)]
    logging.info(f"S3 keys: {s3_keys}")
    out_dir = pathlib.Path(args.out_dir)
    logging.info(f"Output directory: {out_dir}")

    # Prepare work list
    work: list[tuple[str, pathlib.Path, str, bool]] = []
    for key in s3_keys:
        tile = pathlib.Path(key).stem.split("_")[0]
        dst_path = out_dir / f"{tile}_{start_date}_{end_date}.parquet"
        work.append((key, dst_path, args.endpoint_url, args.dry_run))

    # Parallel download
    with _cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_download_single, w): w[0] for w in work}
        for fut in _cf.as_completed(futures):
            msg = fut.result()
            _LOG.info(msg)

    _LOG.info("Finished processing %d file(s).", len(work))


if __name__ == "__main__":  # pragma: no cover
    main()
