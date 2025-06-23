#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import logging
import pathlib
import sys
from typing import Set

import boto3
import botocore
import geopandas as gpd
from shapely.ops import unary_union
from tqdm import tqdm

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
    p.add_argument("--buffer-meters", type=float, default=200, 
                   help="Buffer distance in meters to expand the input geometry")
    p.add_argument("--filter-land-only", action="store_true", 
                   help="Filter parquet files to only include data intersecting with (buffered) land geometries")
    return p.parse_args(argv)


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

    if mgrs_gdf.crs is not None and mgrs_gdf.crs.to_epsg() != 4326:
        _LOG.info(f"Reprojecting MGRS reference layer from {mgrs_gdf.crs} to EPSG:4326.")
        mgrs_gdf = mgrs_gdf.to_crs(4326)
    elif mgrs_gdf.crs is None:
        _LOG.warning("MGRS reference file has no CRS defined. Assuming EPSG:4326.")
        mgrs_gdf.crs = "EPSG:4326" # Assume WGS84 if not set; risky but common for GeoJSONs

    region_gdf = gpd.GeoDataFrame(geometry=region_gs)

    _LOG.info("Performing spatial join between input region and MGRS reference layer...")
    intersecting_mgrs_tiles = gpd.sjoin(mgrs_gdf, region_gdf, how="inner", predicate="intersects")

    if intersecting_mgrs_tiles.empty:
        _LOG.info("No intersecting MGRS tiles found.")
        return set()

    tile_keys = set(intersecting_mgrs_tiles[mgrs_tile_id_column].unique())
    _LOG.info(f"Found {len(tile_keys)} unique intersecting MGRS tile IDs.")
    return tile_keys


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


def _get_utm_crs(lon: float, lat: float) -> str:
    """Get appropriate UTM CRS for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + zone  # Northern hemisphere
    else:
        epsg_code = 32700 + zone  # Southern hemisphere
    return f"EPSG:{epsg_code}"


def _buffer_geometry(geom, buffer_meters: float):
    """Buffer geometry by specified meters using appropriate UTM projection."""
    if buffer_meters <= 0:
        return geom
    
    centroid = geom.centroid
    utm_crs = _get_utm_crs(centroid.x, centroid.y)
    
    gdf_orig = gpd.GeoDataFrame([geom], geometry=[geom], crs="EPSG:4326")
    gdf_projected = gdf_orig.to_crs(utm_crs)
    buffered_geom = gdf_projected.geometry.iloc[0].buffer(buffer_meters)
    
    buffered_gdf = gpd.GeoDataFrame([buffered_geom], geometry=[buffered_geom], crs=utm_crs)
    return buffered_gdf.to_crs(4326).geometry.iloc[0]


def _filter_parquet_by_land(
    parquet_path: pathlib.Path, 
    land_geometry, 
    output_path: pathlib.Path = None
) -> bool:
    """
    Filter a parquet file to only include records that intersect with land geometry.
    
    Returns True if filtering was successful, False if file should be skipped.
    """
    try:
        # Read parquet file
        _LOG.info(f"Filtering {parquet_path.name}...")
        gdf = gpd.read_parquet(parquet_path)
        
        if gdf.empty:
            _LOG.warning(f"Empty parquet file: {parquet_path}")
            return False
        
        # Ensure geometry column exists
        if 'geometry' not in gdf.columns:
            _LOG.error(f"No geometry column found in {parquet_path}")
            return False
        
        # Ensure CRS compatibility
        if gdf.crs is None:
            _LOG.warning(f"No CRS defined for {parquet_path}, assuming EPSG:4326")
            gdf.crs = "EPSG:4326"
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        
        # Create land geometry GeoDataFrame for spatial join with string column name
        land_gdf = gpd.GeoDataFrame({'land_flag': [1]}, geometry=[land_geometry], crs="EPSG:4326")
        
        # Perform spatial filter - keep only intersecting records
        original_count = len(gdf)
        filtered_gdf = gpd.sjoin(gdf, land_gdf, how="inner", predicate="intersects")
        
        # Remove the extra columns from sjoin
        columns_to_drop = []
        if 'index_right' in filtered_gdf.columns:
            columns_to_drop.append('index_right')
        if 'land_flag' in filtered_gdf.columns:
            columns_to_drop.append('land_flag')
        
        if columns_to_drop:
            filtered_gdf = filtered_gdf.drop(columns_to_drop, axis=1)
        
        # Ensure all column names are strings
        filtered_gdf.columns = [str(col) for col in filtered_gdf.columns]
        
        filtered_count = len(filtered_gdf)
        _LOG.info(f"Filtered {parquet_path.name}: {original_count} -> {filtered_count} records "
                 f"({filtered_count/original_count*100:.1f}% retained)")
        
        if filtered_count == 0:
            _LOG.warning(f"No land-intersecting data in {parquet_path.name}, skipping output")
            return False
        
        # Save filtered result
        output_file = output_path or parquet_path
        filtered_gdf.to_parquet(output_file)
        _LOG.info(f"Saved filtered data to {output_file}")
        
        return True
        
    except Exception as e:
        _LOG.error(f"Error filtering {parquet_path}: {e}")
        return False


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

    # Load region
    _LOG.info(f"Loading region from {args.input_file}")
    region = _load_region(args.input_file)
    original_geom = region.iloc[0]
    
    # Buffer if requested
    if args.buffer_meters > 0:
        _LOG.info(f"Applying {args.buffer_meters}m buffer to land geometry")
        buffered_geom = _buffer_geometry(original_geom, args.buffer_meters)
        search_region = gpd.GeoSeries([buffered_geom], crs="EPSG:4326")
    else:
        search_region = region
        buffered_geom = original_geom

    # Get intersecting tiles
    tiles = _get_intersecting_mgrs_ids_from_reference(
        search_region, 
        args.mgrs_reference_file, 
        args.mgrs_tile_id_column
    )
    _LOG.info("Identified %d tile(s).", len(tiles))

    # Build S3 keys & local paths
    s3_keys = [f"{S3_PREFIX}{tile}_{start_date}_{end_date}.parquet" for tile in sorted(tiles)]
    out_dir = pathlib.Path(args.out_dir)

    # Check which files already exist locally
    existing_files = []
    files_to_download = []
    
    for key in s3_keys:
        tile = pathlib.Path(key).stem.split("_")[0]
        dst_path = out_dir / f"{tile}_{start_date}_{end_date}.parquet"
        
        if dst_path.exists() and not args.dry_run:
            existing_files.append(dst_path)
            _LOG.info(f"File already exists: {dst_path.name}")
        else:
            files_to_download.append((key, dst_path, args.endpoint_url, args.dry_run))

    print(f"Found {len(existing_files)} existing files, need to download {len(files_to_download)} files")

    # Download only missing files
    downloaded_files = list(existing_files)  # Start with existing files
    
    if files_to_download:
        _LOG.info(f"Starting downloads of {len(files_to_download)} missing files...")
        print(f"Downloading {len(files_to_download)} missing parquet files...")
        
        with _cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_download_single, w): w[0] for w in files_to_download}
            
            # Add progress bar for downloads
            with tqdm(total=len(futures), desc="Downloading", unit="file") as pbar:
                for fut in _cf.as_completed(futures):
                    msg = fut.result()
                    
                    # Update progress bar description with current file
                    if "Downloaded" in msg:
                        filename = msg.split(" -> ")[-1].split("/")[-1]  # Get just filename
                        pbar.set_description(f"Downloaded {filename}")
                        if not args.dry_run:
                            filepath = msg.split(" -> ")[-1]
                            downloaded_files.append(pathlib.Path(filepath))
                    elif "Missing" in msg:
                        filename = msg.split(" ")[1].split("/")[-1].replace(".parquet", "")
                        pbar.set_description(f"Missing {filename}")
                    elif "dry-run" in msg:
                        filename = msg.split(" ")[1].split("/")[-1].replace(".parquet", "")
                        pbar.set_description(f"Dry-run {filename}")
                    
                    # Log verbose messages only if verbose is enabled
                    if args.verbose:
                        _LOG.info(msg)
                    
                    pbar.update(1)

        print(f"Download complete: {len(downloaded_files) - len(existing_files)} new files downloaded")
    else:
        print("All files already exist locally, skipping downloads")

    print(f"Total files available for processing: {len(downloaded_files)}")
    # Filter parquet files if requested - also add progress bar
    if args.filter_land_only and not args.dry_run:
        _LOG.info("Starting land-only filtering...")
        print(f"Filtering {len(downloaded_files)} files for land-only data...")
        
        filter_geom = buffered_geom
        successful_filters = 0
        
        # Add progress bar for parallel filtering
        with tqdm(total=len(downloaded_files), desc="Filtering", unit="file") as pbar:
            with _cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {}
                for parquet_file in downloaded_files:
                    if parquet_file.exists():
                        futures[ex.submit(_filter_parquet_by_land, parquet_file, filter_geom)] = parquet_file
                
                for fut in _cf.as_completed(futures):
                    parquet_file = futures[fut]
                    filename = parquet_file.name
                    pbar.set_description(f"Filtering {filename}")
                    
                    if fut.result():
                        successful_filters += 1
                    
                    pbar.update(1)
        
        print(f"Filtering complete: {successful_filters}/{len(downloaded_files)} files retained data")
        _LOG.info(f"Successfully filtered {successful_filters}/{len(downloaded_files)} parquet files")

    _LOG.info("Finished processing.")


if __name__ == "__main__":  # pragma: no cover
    main()
