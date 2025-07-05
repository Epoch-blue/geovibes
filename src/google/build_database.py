import argparse
import logging
import pathlib
import sys
import time
import tempfile
import subprocess
import shutil
from typing import List, Optional, Dict, Tuple

import geopandas as gpd
import duckdb
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm
from joblib import Parallel, delayed
import os


def setup_logging() -> None:
    """Configure basic logging settings for the database build process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def is_cloud_path(path: str) -> bool:
    """
    Determine if a file path points to cloud storage.
    
    Args:
        path: File system path to check
        
    Returns:
        True if path is a cloud storage URL (GCS or S3), False otherwise
    """
    return path.startswith('gs://') or path.startswith('s3://')


def upload_to_cloud(local_file: str, cloud_path: str) -> bool:
    """
    Upload a local file to cloud storage using appropriate CLI tools.
    
    Args:
        local_file: Path to local file to upload
        cloud_path: Target cloud storage path (gs:// or s3://)
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        if cloud_path.startswith('gs://'):
            result = subprocess.run(['gsutil', 'cp', local_file, cloud_path], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to upload {local_file} to {cloud_path}: {result.stderr}")
                return False
            logging.info(f"Uploaded {local_file} to {cloud_path}")
            return True
        elif cloud_path.startswith('s3://'):
            result = subprocess.run(['aws', 's3', 'cp', local_file, cloud_path], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to upload {local_file} to {cloud_path}: {result.stderr}")
                return False
            logging.info(f"Uploaded {local_file} to {cloud_path}")
            return True
        else:
            logging.error(f"Unsupported cloud path: {cloud_path}")
            return False
    except Exception as e:
        logging.error(f"Error uploading {local_file} to {cloud_path}: {e}")
        return False


def ensure_cloud_directory(cloud_dir_path: str) -> bool:
    """
    Ensure cloud directory exists. For cloud storage, directories are implicit.
    
    Args:
        cloud_dir_path: Cloud storage directory path
        
    Returns:
        True if directory handling successful, False otherwise
    """
    try:
        if cloud_dir_path.startswith('gs://') or cloud_dir_path.startswith('s3://'):
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error ensuring cloud directory {cloud_dir_path}: {e}")
        return False


def check_cloud_file_exists(cloud_path: str) -> bool:
    """
    Check if a file exists in cloud storage.
    
    Args:
        cloud_path: Cloud storage file path (gs:// or s3://)
        
    Returns:
        True if file exists, False otherwise
    """
    try:
        if cloud_path.startswith('gs://'):
            result = subprocess.run(['gsutil', 'ls', cloud_path], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        elif cloud_path.startswith('s3://'):
            result = subprocess.run(['aws', 's3', 'ls', cloud_path], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        else:
            return False
    except Exception:
        return False


def check_single_file_status(mgrs_id: str, epsg_code: str, working_output_dir: str, 
                            output_dir_is_cloud: bool, output_dir: str, gcs_bucket: str) -> Tuple[str, Optional[str], Optional[Tuple[str, str, str]]]:
    """
    Check status of a single MGRS tile file in parallel.
    
    Args:
        mgrs_id: MGRS tile identifier
        epsg_code: EPSG code for the tile
        working_output_dir: Local working directory
        output_dir_is_cloud: Whether output directory is cloud storage
        output_dir: Output directory path
        gcs_bucket: GCS bucket name for source files
        
    Returns:
        Tuple of (mgrs_id, existing_file_path, processing_task)
        - existing_file_path: Path to existing file (local or cloud), None if not found
        - processing_task: Task tuple for processing if file doesn't exist, None if file exists
    """
    expected_filename = f"{mgrs_id}_2024.parquet"
    local_path = pathlib.Path(working_output_dir) / expected_filename
    
    # Check local file first
    if local_path.exists():
        return mgrs_id, str(local_path), None
    
    # If output dir is cloud, check for existing processed file in cloud
    if output_dir_is_cloud:
        cloud_parquet_path = f"{output_dir.rstrip('/')}/{expected_filename}"
        if check_cloud_file_exists(cloud_parquet_path):
            return mgrs_id, cloud_parquet_path, None
    
    # File doesn't exist, needs processing
    gcs_path = f"gs://{gcs_bucket}/embeddings/google_satellite_v1/25_0_10/{mgrs_id}_2024.geojson"
    return mgrs_id, None, (gcs_path, epsg_code, working_output_dir)




def _load_region(path: str) -> gpd.GeoSeries:
    """
    Load a vector file and return a single unified geometry in WGS84.
    
    Args:
        path: Path to vector file (GeoJSON, Shapefile, etc.)
        
    Returns:
        GeoSeries containing unified geometry in WGS84 coordinate system
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If no geometries found in the input file
    """
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("No geometries found in input file")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")


def _get_utm_crs(lon: float, lat: float) -> str:
    """
    Get appropriate UTM CRS EPSG code for given coordinates.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        EPSG code string for the appropriate UTM zone
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"

def get_intersecting_mgrs_ids(roi_path: str, mgrs_reference_path: str) -> Dict[str, str]:
    """
    Find MGRS tiles that intersect with the ROI and return mapping of tile_id to EPSG code.
    
    This function is critical for the database build workflow as it determines which
    satellite imagery tiles need to be processed based on the region of interest.
    
    Args:
        roi_path: Path to ROI vector file (GeoJSON, Shapefile, etc.)
        mgrs_reference_path: Path to MGRS grid reference file containing tile boundaries
        
    Returns:
        Dictionary mapping MGRS tile IDs to their corresponding EPSG codes
        
    Raises:
        ValueError: If required columns are missing from MGRS reference file
    """
    logging.info("Loading ROI and MGRS reference file...")
    region_gs = _load_region(roi_path)
    mgrs_gdf = gpd.read_parquet(mgrs_reference_path)

    if mgrs_gdf.crs is None or mgrs_gdf.crs.to_epsg() != 4326:
        mgrs_gdf = mgrs_gdf.to_crs(4326)

    logging.info("Finding intersecting MGRS tiles...")
    intersecting_tiles = gpd.sjoin(
        mgrs_gdf, gpd.GeoDataFrame(geometry=region_gs), how="inner", predicate="intersects"
    )

    tile_id_col = "mgrs_id"
    epsg_col = "epsg"
    if tile_id_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have a '{tile_id_col}' column.")
    if epsg_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have an '{epsg_col}' column.")

    return dict(zip(intersecting_tiles[tile_id_col], intersecting_tiles[epsg_col]))

def process_and_save_geojson(gcs_path: str, epsg_code: str, output_dir: str) -> Optional[str]:
    """
    Process a single GeoJSON file from GCS and save as GeoParquet.
    
    This function transforms satellite imagery embeddings from GeoJSON format
    into optimized GeoParquet format for efficient database ingestion. It handles
    coordinate transformation, centroid calculation, and embedding preparation.
    
    Args:
        gcs_path: Path to GeoJSON file in Google Cloud Storage
        epsg_code: EPSG code for the tile's UTM projection
        output_dir: Local directory to save the processed GeoParquet file
        
    Returns:
        Path to saved GeoParquet file, or None if processing failed
    """
    try:
        output_filename = f"{pathlib.Path(gcs_path).stem}.parquet"
        local_path = os.path.join(output_dir, output_filename)
            
        gdf = gpd.read_file(gcs_path).drop(columns=["id"])
        if gdf.empty:
            return None

        centroids = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                centroids.append(None)
                continue
            
            centroid_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(f"EPSG:{epsg_code}").centroid
            centroid_wgs84 = centroid_utm.to_crs("EPSG:4326").iloc[0]
            centroids.append(centroid_wgs84)

        gdf['geometry'] = centroids
        gdf = gdf[gdf.geometry.notna()]
        
        band_names = [f'A{i:02d}' for i in range(64)]
        
        for band in band_names:
            if band not in gdf.columns:
                gdf[band] = 0
                
        band_subset = gdf[band_names]
        embedding_data = band_subset.values
        gdf['embedding'] = embedding_data.tolist()

        if 'id' in gdf.columns:
            gdf = gdf.rename(columns={'id': 'tile_id'})
        elif 'tile_id' not in gdf.columns:
            mgrs_id = pathlib.Path(gcs_path).stem.split('_')[0]
            gdf['tile_id'] = mgrs_id
        
        final_cols = ['tile_id', 'geometry', 'embedding']
        final_gdf: gpd.GeoDataFrame = gdf[final_cols]
        
        final_gdf.to_parquet(local_path)
        
        return local_path
    
    except Exception as e:
        logging.warning(f"Failed to process {gcs_path}: {e}")
        return None

def check_and_clean_embeddings(parquet_files: List[str]) -> List[str]:
    """
    Check parquet files for NaN embeddings and create cleaned versions.
    
    This function ensures data quality by identifying and removing rows with
    invalid embeddings before database ingestion. It's essential for maintaining
    consistent vector similarity search performance.
    
    Args:
        parquet_files: List of paths to parquet files to check
        
    Returns:
        List of paths to cleaned parquet files (original or cleaned versions)
    """
    
    logging.info("Checking parquet files for NaN values in embeddings...")
    cleaned_files = []
    total_nan_count = 0
    files_with_nans = []
    
    for parquet_file in tqdm(parquet_files, desc="Checking parquet files"):
        try:
            df = gpd.read_parquet(parquet_file)
        except Exception:
            df = pd.read_parquet(parquet_file)
        original_count = len(df)
        
        if 'embedding' in df.columns:
            embeddings = np.array(df['embedding'].tolist())
            
            nan_mask = np.isnan(embeddings).any(axis=1)
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                logging.warning(f"Found {nan_count} embeddings with NaN values in {pathlib.Path(parquet_file).name}")
                files_with_nans.append(pathlib.Path(parquet_file).name)
                total_nan_count += nan_count
                
                df_clean = df[~nan_mask].copy()
                
                clean_filename = parquet_file.replace('.parquet', '_clean.parquet')
                if isinstance(df_clean, gpd.GeoDataFrame):
                    df_clean.to_parquet(clean_filename, index=False)
                else:
                    if 'geometry' in df_clean.columns:
                        df_clean = gpd.GeoDataFrame(df_clean)
                    df_clean.to_parquet(clean_filename, index=False)
                cleaned_files.append(clean_filename)
                
                logging.info(f"Cleaned {pathlib.Path(parquet_file).name}: {original_count} -> {len(df_clean)} rows")
            else:
                cleaned_files.append(parquet_file)
        else:
            logging.warning(f"No 'embedding' column found in {pathlib.Path(parquet_file).name}")
            cleaned_files.append(parquet_file)
    
    if total_nan_count > 0:
        logging.warning(f"NaN SUMMARY:")
        logging.warning(f"  Total embeddings with NaN values: {total_nan_count}")
        logging.warning(f"  Files affected: {len(files_with_nans)}")
        logging.warning(f"  Affected files: {', '.join(files_with_nans)}")
        logging.warning(f"  Cleaned versions created with '_clean.parquet' suffix")
    else:
        logging.info("✅ No NaN values found in any embeddings")
    
    return cleaned_files


def create_duckdb_index(
    parquet_files: List[str], 
    output_file: str, 
    metric: str,
    embedding_column: str = "embedding",
    embedding_dim: int = 64
) -> None:
    """
    Create a DuckDB database with HNSW and spatial indexes from parquet files.
    
    This is the final step of the database build process, creating an optimized
    database with vector similarity search capabilities and spatial indexing
    for efficient querying of satellite imagery embeddings.
    
    Args:
        parquet_files: List of paths to parquet files containing embeddings
        output_file: Path where the DuckDB database will be saved
        metric: Distance metric for HNSW index ('cosine', 'l2sq', 'inner_product')
        embedding_column: Name of the column containing embedding vectors
        embedding_dim: Dimension of the embedding vectors
    """
    
    cleaned_parquet_files = check_and_clean_embeddings(parquet_files)
    
    logging.info(f"Creating DuckDB database at {output_file} with {metric} metric...")
    logging.info(f"Using embedding dimension: {embedding_dim}")

    con = duckdb.connect(database=output_file)

    logging.info("Loading extensions...")
    con.execute("SET enable_progress_bar=true")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("INSTALL httpfs; LOAD httpfs;")  # Required for GCS/S3 access
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    
    # Set up GCS authentication if we have cloud files
    has_cloud_files = any(is_cloud_path(f) for f in cleaned_parquet_files)
    if has_cloud_files:
        logging.info("Setting up cloud storage authentication...")
        # Try to use environment variables for GCS authentication
        gcs_access_key = os.environ.get('GCS_ACCESS_KEY_ID')
        gcs_secret_key = os.environ.get('GCS_SECRET_ACCESS_KEY')
        
        if gcs_access_key and gcs_secret_key:
            logging.info("Using HMAC keys for GCS authentication")
            con.execute(f"""
                CREATE SECRET (
                    TYPE gcs,
                    KEY_ID '{gcs_access_key}',
                    SECRET '{gcs_secret_key}'
                );
            """)
        else:
            logging.info("Using default GCS authentication (gcloud credentials)")
            # DuckDB will use default Google Cloud credentials

    path_strings = []
    for p in cleaned_parquet_files:
        if is_cloud_path(p):
            # Cloud paths should be used as-is, not resolved as local paths
            path_strings.append(f"'{p}'")
        else:
            # Local paths need to be resolved to absolute paths
            path_str = str(pathlib.Path(p).resolve()).replace("\\", "/")
            path_strings.append(f"'{path_str}'")
    
    sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"

    create_table_sql = f"""
    CREATE OR REPLACE TABLE geo_embeddings AS
    SELECT
        tile_id AS id,
        CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
        geometry
    FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
    """

    logging.info("Creating table and ingesting data...")
    start_time = time.time()
    con.execute(create_table_sql)
    ingest_time = time.time() - start_time

    result = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()
    if result is None:
        logging.error("Failed to get row count from database")
        con.close()
        return
    row_count = result[0]
    logging.info(f"Ingested {row_count} rows in {ingest_time:.2f} seconds")

    if row_count > 0:
        logging.info(f"Creating HNSW index with {metric} metric...")
        start_index_time = time.time()
        con.execute(
            f"CREATE INDEX IF NOT EXISTS emb_hnsw_idx ON geo_embeddings USING HNSW (embedding) WITH (metric = '{metric}');"
        )
        index_time = time.time() - start_index_time
        logging.info(f"HNSW index created in {index_time:.2f} seconds")

        logging.info("Creating RTree spatial index...")
        start_spatial_index_time = time.time()
        con.execute(
            "CREATE INDEX IF NOT EXISTS geom_spatial_idx ON geo_embeddings USING RTREE (geometry);"
        )
        spatial_index_time = time.time() - start_spatial_index_time
        logging.info(f"RTree spatial index created in {spatial_index_time:.2f} seconds")

        db_size_info = con.execute("PRAGMA database_size;").fetchone()
        logging.info(f"Database size: {db_size_info}")

    con.close()
    logging.info(f"Database saved to {output_file}")

def main() -> None:
    """
    Main orchestration function for building satellite imagery embedding databases.
    
    This function coordinates the entire database build workflow:
    1. Parse command-line arguments and set up logging
    2. Determine MGRS tiles that intersect with the ROI
    3. Check for existing processed files (local or cloud) to avoid reprocessing
    4. Process missing GeoJSON files in parallel and convert to GeoParquet
    5. Build DuckDB database with HNSW and spatial indexes (mixing cloud and local paths)
    6. Upload new results to cloud storage if specified
    7. Clean up temporary files
    
    The function efficiently handles cloud storage by:
    - Reading existing parquet files directly from cloud storage via DuckDB
    - Only processing missing files locally
    - Uploading only newly processed files
    - Avoiding unnecessary downloads and uploads
    """
    parser = argparse.ArgumentParser(description="Process Google Satellite GeoJSON files and create a DuckDB index.")
    parser.add_argument("roi_file", help="Path to the ROI file (e.g., aoi.geojson).")
    parser.add_argument("output_dir", help="Directory to save output files (local path or gs://bucket/path or s3://bucket/path).")
    parser.add_argument("output_db_dir", help="Directory for the output DuckDB database file (local path or gs://bucket/path or s3://bucket/path).")
    parser.add_argument("--mgrs_reference_file", default="gs://geovibes/geometries/mgrs_tiles.parquet", help="Path to the MGRS grid reference file.")
    parser.add_argument("--gcs_bucket", default="geovibes", help="GCS bucket to use for the embeddings.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of the embedding vectors.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers for processing files.")
    args = parser.parse_args()

    setup_logging()

    # Extract region name from ROI file and construct database filename
    # Handle both local and cloud paths
    if is_cloud_path(args.roi_file):
        # For cloud paths, extract filename manually
        roi_basename = args.roi_file.rstrip('/').split('/')[-1].rsplit('.', 1)[0]
    else:
        roi_basename = pathlib.Path(args.roi_file).stem
    
    db_filename = f"{roi_basename}_google.db"
    
    # Construct full output database path
    if is_cloud_path(args.output_db_dir):
        args.output_db_file = f"{args.output_db_dir.rstrip('/')}/{db_filename}"
    else:
        args.output_db_file = str(pathlib.Path(args.output_db_dir) / db_filename)
    
    logging.info(f"Database will be saved as: {args.output_db_file}")

    output_dir_is_cloud = is_cloud_path(args.output_dir)
    output_db_is_cloud = is_cloud_path(args.output_db_file)
    
    if output_dir_is_cloud:
        temp_dir = tempfile.mkdtemp(prefix='build_database_')
        working_output_dir = temp_dir
        logging.info(f"Using temporary directory {working_output_dir} for cloud output {args.output_dir}")
        ensure_cloud_directory(args.output_dir)
    else:
        working_output_dir = args.output_dir
        output_path = pathlib.Path(working_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    if output_db_is_cloud:
        temp_db_fd, working_db_file = tempfile.mkstemp(suffix='.db', prefix='build_database_')
        os.close(temp_db_fd)
        os.unlink(working_db_file)  # Delete the empty file so DuckDB can create a fresh database
        logging.info(f"Using temporary database file {working_db_file} for cloud output {args.output_db_file}")
    else:
        working_db_file = args.output_db_file
        db_parent = pathlib.Path(working_db_file).parent
        db_parent.mkdir(parents=True, exist_ok=True)

    try:
        mgrs_epsg_map = get_intersecting_mgrs_ids(args.roi_file, args.mgrs_reference_file)
        if not mgrs_epsg_map:
            logging.info("No intersecting MGRS tiles found for the given ROI.")
            return

        local_parquet_files = []
        tasks_to_process = []

        logging.info("Checking for existing parquet files in parallel...")
        
        # Prepare tasks for parallel file existence checking
        file_check_tasks = [
            (mgrs_id, epsg_code, working_output_dir, output_dir_is_cloud, args.output_dir, args.gcs_bucket)
            for mgrs_id, epsg_code in mgrs_epsg_map.items()
        ]
        
        # Check file existence in parallel
        file_status_results_raw = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(check_single_file_status)(*task)
            for task in file_check_tasks
        )
        file_status_results = list(file_status_results_raw) if file_status_results_raw is not None else []
        
        # Process results
        cloud_files_found = 0
        local_files_found = 0
        
        for mgrs_id, existing_file_path, processing_task in file_status_results:
            if existing_file_path:
                local_parquet_files.append(existing_file_path)
                if is_cloud_path(existing_file_path):
                    cloud_files_found += 1
                    logging.info(f"Found existing processed file in cloud: {existing_file_path}")
                else:
                    local_files_found += 1
            elif processing_task is not None:
                tasks_to_process.append(processing_task)
        
        existing_files_msg = f"Found {len(local_parquet_files)} existing parquet files"
        if output_dir_is_cloud:
            cloud_files = [f for f in local_parquet_files if is_cloud_path(f)]
            local_files = [f for f in local_parquet_files if not is_cloud_path(f)]
            if cloud_files:
                existing_files_msg += f" ({len(cloud_files)} in cloud, {len(local_files)} local)"
        logging.info(existing_files_msg)
        
        if tasks_to_process:
            logging.info(f"Constructed {len(tasks_to_process)} tasks to process for missing files.")

            processed_files_result = Parallel(n_jobs=args.workers, verbose=20)(
                delayed(process_and_save_geojson)(*task)
                for task in tqdm(tasks_to_process, desc="Processing GeoJSON files")
            )
            processed_files = list(processed_files_result)

            newly_processed_files = [f for f in processed_files if f is not None]
            failed_downloads = len(processed_files) - len(newly_processed_files)
            
            if failed_downloads > 0:
                logging.error(f"Failed to download/process {failed_downloads} out of {len(tasks_to_process)} required files.")
                logging.error("Cannot proceed with incomplete data. All required files must be successfully downloaded.")
                sys.exit(1)
            
            local_parquet_files.extend(newly_processed_files)
            logging.info(f"Successfully processed all {len(newly_processed_files)} missing files.")

        expected_file_count = len(mgrs_epsg_map)
        actual_file_count = len(local_parquet_files)
        
        if actual_file_count != expected_file_count:
            logging.error(f"File count mismatch: Expected {expected_file_count} files but have {actual_file_count} files.")
            logging.error("Cannot proceed with incomplete data.")
            sys.exit(1)

        if not local_parquet_files:
            logging.error("No data could be processed or found.")
            sys.exit(1)

        logging.info(f"Validated all {len(local_parquet_files)} required parquet files are present.")
        
        create_duckdb_index(local_parquet_files, working_db_file, args.metric, embedding_dim=args.embedding_dim)

        upload_success = True
        
        if output_dir_is_cloud:
            # Only upload newly processed files (local paths), not existing cloud files
            local_files_to_upload = [f for f in local_parquet_files if not is_cloud_path(f)]
            if local_files_to_upload:
                logging.info(f"Uploading {len(local_files_to_upload)} newly processed parquet files to cloud directory: {args.output_dir}")
                for parquet_file in local_files_to_upload:
                    filename = pathlib.Path(parquet_file).name
                    cloud_file_path = f"{args.output_dir.rstrip('/')}/{filename}"
                    if not upload_to_cloud(parquet_file, cloud_file_path):
                        upload_success = False
                        break
            else:
                logging.info("No new parquet files to upload - all files already exist in cloud directory")
        
        if output_db_is_cloud and upload_success:
            logging.info(f"Uploading database file to cloud: {args.output_db_file}")
            if not upload_to_cloud(working_db_file, args.output_db_file):
                upload_success = False
        
        if not upload_success:
            logging.error("Failed to upload files to cloud storage")
            sys.exit(1)
        
        if output_dir_is_cloud and tasks_to_process:
            # Only clean up if we actually used the temporary directory for processing
            logging.info(f"Cleaning up temporary directory: {working_output_dir}")
            shutil.rmtree(working_output_dir, ignore_errors=True)
        
        if output_db_is_cloud:
            logging.info(f"Cleaning up temporary database file: {working_db_file}")
            try:
                os.unlink(working_db_file)
            except Exception:
                pass
        
        if output_dir_is_cloud or output_db_is_cloud:
            logging.info("All files successfully uploaded to cloud storage")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        
        if 'working_output_dir' in locals() and output_dir_is_cloud:
            shutil.rmtree(working_output_dir, ignore_errors=True)
        if 'working_db_file' in locals() and output_db_is_cloud:
            try:
                os.unlink(working_db_file)
            except Exception:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()