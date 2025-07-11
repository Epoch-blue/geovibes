import argparse
import logging
import pathlib
import time
from typing import List, Optional
import os
import tempfile
import shutil

import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from google.cloud import storage


def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_and_clean_embeddings(parquet_files: List[str]) -> List[str]:
    """Check parquet files for NaN embeddings and create cleaned versions."""
    
    logging.info("Checking parquet files for NaN values in embeddings...")
    cleaned_files = []
    total_nan_count = 0
    files_with_nans = []
    
    for parquet_file in tqdm(parquet_files, desc="Checking parquet files"):
        # Try reading as regular parquet first (faster)
        try:
            df = pd.read_parquet(parquet_file)
            # If it has geometry-related columns, try to read as GeoDataFrame
            if any(col in df.columns for col in ['geometry', 'geometry_wkt', 'geom', 'wkt']):
                try:
                    df = gpd.read_parquet(parquet_file)
                except:
                    # Keep as regular dataframe if geopandas fails
                    pass
        except Exception as e:
            logging.error(f"Failed to read {parquet_file}: {e}")
            continue
            
        original_count = len(df)
        
        # Check for NaN values in embedding column
        if 'embedding' in df.columns:
            # Convert embeddings to numpy arrays for NaN checking
            embeddings = np.array(df['embedding'].tolist())
            
            # Find rows with any NaN values in embeddings
            nan_mask = np.isnan(embeddings).any(axis=1)
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                logging.warning(f"Found {nan_count} embeddings with NaN values in {pathlib.Path(parquet_file).name}")
                files_with_nans.append(pathlib.Path(parquet_file).name)
                total_nan_count += nan_count
                
                # Remove rows with NaN embeddings
                df_clean = df[~nan_mask].copy()
                
                # Save cleaned version
                clean_filename = parquet_file.replace('.parquet', '_clean.parquet')
                if isinstance(df_clean, gpd.GeoDataFrame):
                    df_clean.to_parquet(clean_filename, index=False)
                else:
                    # Just save as regular parquet
                    df_clean.to_parquet(clean_filename, index=False)
                cleaned_files.append(clean_filename)
                
                logging.info(f"Cleaned {pathlib.Path(parquet_file).name}: {original_count} -> {len(df_clean)} rows")
            else:
                # No NaN values, use original file
                cleaned_files.append(parquet_file)
        else:
            logging.warning(f"No 'embedding' column found in {pathlib.Path(parquet_file).name}")
            cleaned_files.append(parquet_file)
    
    # Summary report
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
    metric: str = "cosine",
    embedding_column: str = "embedding",
    id_column: str = "id",
    geometry_column: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    skip_nan_check: bool = False
) -> None:
    """Creates a DuckDB database with HNSW and spatial indexes from parquet files.
    
    Args:
        parquet_files: List of parquet file paths
        output_file: Output database file path
        metric: Distance metric for HNSW index
        embedding_column: Name of the embedding column
        id_column: Name of the ID column
        geometry_column: Name of the geometry column (auto-detected if None)
        embedding_dim: Dimension of embeddings (auto-detected if None)
        skip_nan_check: Skip NaN checking and cleaning
    """
    
    if not skip_nan_check:
        cleaned_parquet_files = check_and_clean_embeddings(parquet_files)
    else:
        cleaned_parquet_files = parquet_files
    
    # Auto-detect embedding dimension if not provided
    if embedding_dim is None:
        logging.info("Auto-detecting embedding dimension...")
        sample_df = pd.read_parquet(cleaned_parquet_files[0], columns=[embedding_column])
        if embedding_column in sample_df.columns and len(sample_df) > 0:
            first_embedding = sample_df[embedding_column].iloc[0]
            if isinstance(first_embedding, list):
                embedding_dim = len(first_embedding)
            else:
                embedding_dim = len(first_embedding)
            logging.info(f"Detected embedding dimension: {embedding_dim}")
        else:
            raise ValueError(f"Could not detect embedding dimension from column '{embedding_column}'")
    
    # Auto-detect geometry column if not specified
    if geometry_column is None:
        logging.info("Auto-detecting geometry column...")
        sample_df = pd.read_parquet(cleaned_parquet_files[0])
        geometry_candidates = ['geometry', 'geometry_wkt', 'geom', 'wkt', 'geo', 'shape']
        for col in geometry_candidates:
            if col in sample_df.columns:
                geometry_column = col
                logging.info(f"Detected geometry column: {geometry_column}")
                break
        
        if geometry_column is None:
            logging.error("CRITICAL: No geometry column found in parquet files!")
            logging.error(f"Checked for columns: {geometry_candidates}")
            logging.error(f"Available columns: {list(sample_df.columns)}")
            raise ValueError("Geometry column is required for GeoVibes databases. "
                           "Please specify --geometry-column or ensure your parquet files contain a geometry column.")
    
    logging.info(f"Creating DuckDB database at {output_file} with {metric} metric...")
    logging.info(f"Using embedding dimension: {embedding_dim}")
    logging.info(f"ID column: {id_column}, Geometry column: {geometry_column}")

    con = duckdb.connect(database=output_file)

    logging.info("Loading extensions...")
    con.execute("SET enable_progress_bar=true")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")

    # Convert string paths to Path objects and create proper path strings
    path_strings = [
        f"'{p_str}'"
        for p_str in (str(pathlib.Path(p).resolve()).replace("\\", "/") for p in cleaned_parquet_files)
    ]
    sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"

    # Build the CREATE TABLE SQL - geometry is always required
    # Check if it's a WKT column that needs conversion
    sample_df = pd.read_parquet(cleaned_parquet_files[0])
    is_wkt_column = geometry_column in ['geometry_wkt', 'wkt', 'geo_wkt'] or \
                   (geometry_column in sample_df.columns and 
                    len(sample_df) > 0 and
                    isinstance(sample_df[geometry_column].iloc[0], str) and 
                    sample_df[geometry_column].iloc[0].startswith(('POINT', 'POLYGON', 'LINESTRING')))
    
    if is_wkt_column:
        # Need to convert WKT to geometry
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            {id_column} AS id,
            CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
            ST_GeomFromText({geometry_column}) as geometry
        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
        """
    else:
        # Direct geometry column
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            {id_column} AS id,
            CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
            {geometry_column} as geometry
        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
        """

    logging.info("Creating table and ingesting data...")
    start_time = time.time()
    con.execute(create_table_sql)
    ingest_time = time.time() - start_time

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
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


def upload_to_gcs(local_file: str, gcs_path: str) -> None:
    """Upload a file to Google Cloud Storage."""
    if not gcs_path.startswith('gs://'):
        raise ValueError("GCS path must start with 'gs://'")
    
    # Parse bucket and blob path
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else os.path.basename(local_file)
    
    logging.info(f"Uploading {local_file} to gs://{bucket_name}/{blob_name}...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(local_file)
    logging.info(f"Upload complete: {gcs_path}")


def download_gcs_files(gcs_paths: List[str], temp_dir: str) -> List[str]:
    """Download parquet files from GCS to a temporary directory."""
    storage_client = storage.Client()
    local_paths = []
    
    for gcs_path in tqdm(gcs_paths, desc="Downloading files from GCS"):
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        # Parse bucket and blob
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        # Download to temp directory
        local_filename = os.path.join(temp_dir, os.path.basename(blob_name))
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_filename)
        local_paths.append(local_filename)
        
    return local_paths


def list_gcs_parquet_files(gcs_path: str) -> List[str]:
    """List all parquet files in a GCS directory."""
    if not gcs_path.startswith('gs://'):
        raise ValueError("GCS path must start with 'gs://'")
    
    # Parse bucket and prefix
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    parquet_files = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('.parquet'):
            parquet_files.append(f"gs://{bucket_name}/{blob.name}")
    
    return parquet_files


def main():
    parser = argparse.ArgumentParser(description="Create a DuckDB database with HNSW and spatial indexes from parquet files.")
    parser.add_argument("parquet_files", nargs='+', help="Paths to parquet files or directories (local or gs://)")
    parser.add_argument("output_path", help="Output path for the DuckDB database (local path or gs:// path)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index")
    parser.add_argument("--embedding-column", default="embedding", help="Name of the embedding column")
    parser.add_argument("--id-column", default="id", help="Name of the ID column")
    parser.add_argument("--geometry-column", help="Name of the geometry column (auto-detected if not specified, required for database creation)")
    parser.add_argument("--embedding-dim", type=int, help="Dimension of embeddings (auto-detected if not specified)")
    parser.add_argument("--skip-nan-check", action="store_true", help="Skip NaN checking and cleaning")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Collect all parquet files
    all_parquet_files = []
    gcs_files = []
    
    for path in args.parquet_files:
        if path.startswith('gs://'):
            # Handle GCS paths
            if path.endswith('.parquet'):
                # Single file
                gcs_files.append(path)
            else:
                # Directory - list all parquet files
                gcs_parquet_files = list_gcs_parquet_files(path)
                gcs_files.extend(gcs_parquet_files)
                logging.info(f"Found {len(gcs_parquet_files)} parquet files in {path}")
        else:
            # Handle local paths
            path_obj = pathlib.Path(path)
            if path_obj.is_file() and path_obj.suffix == '.parquet':
                all_parquet_files.append(str(path_obj))
            elif path_obj.is_dir():
                # Find all parquet files in directory
                parquet_files = list(path_obj.glob("*.parquet"))
                all_parquet_files.extend([str(f) for f in parquet_files])
            else:
                logging.warning(f"Skipping {path} - not a parquet file or directory")
    
    # Download GCS files if any
    temp_dir = None
    if gcs_files:
        temp_dir = tempfile.mkdtemp(prefix="geovibes_db_")
        logging.info(f"Downloading {len(gcs_files)} files from GCS to temporary directory...")
        local_gcs_files = download_gcs_files(gcs_files, temp_dir)
        all_parquet_files.extend(local_gcs_files)
    
    if not all_parquet_files:
        logging.error("No parquet files found!")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return
    
    logging.info(f"Total {len(all_parquet_files)} parquet files to process")
    
    # Determine if output is GCS or local
    if args.output_path.startswith('gs://'):
        # Create local temp file first
        local_temp_file = "temp_database.db"
        create_duckdb_index(
            all_parquet_files,
            local_temp_file,
            metric=args.metric,
            embedding_column=args.embedding_column,
            id_column=args.id_column,
            geometry_column=args.geometry_column,
            embedding_dim=args.embedding_dim,
            skip_nan_check=args.skip_nan_check
        )
        
        # Upload to GCS
        upload_to_gcs(local_temp_file, args.output_path)
        
        # Clean up temp file
        os.remove(local_temp_file)
        logging.info(f"Cleaned up temporary file: {local_temp_file}")
    else:
        # Direct local output
        create_duckdb_index(
            all_parquet_files,
            args.output_path,
            metric=args.metric,
            embedding_column=args.embedding_column,
            id_column=args.id_column,
            geometry_column=args.geometry_column,
            embedding_dim=args.embedding_dim,
            skip_nan_check=args.skip_nan_check
        )
    
    # Clean up temporary directory if used
    if temp_dir:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()