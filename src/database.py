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
import fsspec
from joblib import Parallel, delayed


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_and_clean_embeddings(parquet_files: List[str]) -> List[str]:
    
    logging.info("Checking parquet files for NaN values in embeddings...")
    cleaned_files = []
    total_nan_count = 0
    files_with_nans = []
    
    for parquet_file in tqdm(parquet_files, desc="Checking parquet files"):
        try:
            df = pd.read_parquet(parquet_file)
            if any(col in df.columns for col in ['geometry', 'geometry_wkt', 'geom', 'wkt']):
                try:
                    df = gpd.read_parquet(parquet_file)
                except:
                    pass
        except Exception as e:
            logging.error(f"Failed to read {parquet_file}: {e}")
            continue
            
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
    metric: str = "cosine",
    embedding_column: str = "embedding",
    id_column: str = "id",
    geometry_column: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    skip_nan_check: bool = False
) -> None:
    
    if not skip_nan_check:
        cleaned_parquet_files = check_and_clean_embeddings(parquet_files)
    else:
        cleaned_parquet_files = parquet_files
    
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

    path_strings = [
        f"'{p_str}'"
        for p_str in (str(pathlib.Path(p).resolve()).replace("\\", "/") for p in cleaned_parquet_files)
    ]
    sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"

    sample_df = pd.read_parquet(cleaned_parquet_files[0])
    is_wkt_column = geometry_column in ['geometry_wkt', 'wkt', 'geo_wkt'] or \
                   (geometry_column in sample_df.columns and 
                    len(sample_df) > 0 and
                    isinstance(sample_df[geometry_column].iloc[0], str) and 
                    sample_df[geometry_column].iloc[0].startswith(('POINT', 'POLYGON', 'LINESTRING')))
    
    if is_wkt_column:
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            {id_column} AS id,
            CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
            ST_GeomFromText({geometry_column}) as geometry
        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
        """
    else:
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

    row_count_result = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()
    row_count = row_count_result[0] if row_count_result else 0
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


def get_cloud_protocol(path: str) -> Optional[str]:
    if path.startswith("s3://"):
        return "s3"
    if path.startswith("gs://"):
        return "gs"
    return None


def upload_to_cloud(local_file: str, cloud_path: str) -> None:
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")
    
    logging.info(f"Uploading {local_file} to {cloud_path}...")
    fs = fsspec.filesystem(protocol)
    fs.put(local_file, cloud_path)
    logging.info(f"Upload complete: {cloud_path}")


def _download_single_cloud_file(cloud_path: str, temp_dir: str) -> Optional[str]:
    try:
        protocol = get_cloud_protocol(cloud_path)
        if not protocol:
            logging.error(f"Invalid cloud path provided to worker: {cloud_path}")
            return None
        
        local_filename = os.path.join(temp_dir, os.path.basename(cloud_path))

        if os.path.exists(local_filename):
            return local_filename

        fs = fsspec.filesystem(protocol)
        fs.get(cloud_path, local_filename)
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {cloud_path} in worker: {e}")
        return None


def download_cloud_files(cloud_paths: List[str], temp_dir: str) -> List[str]:
    logging.info(f"Downloading {len(cloud_paths)} files in parallel using joblib...")
    
    local_paths = Parallel(n_jobs=-1)(
        delayed(_download_single_cloud_file)(cloud_path, temp_dir)
        for cloud_path in tqdm(cloud_paths, desc="Queueing cloud downloads")
    )
    
    successful_downloads = [path for path in local_paths if path is not None]
    
    if len(successful_downloads) != len(cloud_paths):
        logging.warning(
            f"Failed to download {len(cloud_paths) - len(successful_downloads)} files. "
            "Check logs for errors."
        )

    return successful_downloads


def list_cloud_parquet_files(cloud_path: str) -> List[str]:
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")

    if not cloud_path.endswith('/'):
        cloud_path += '/'
    
    fs = fsspec.filesystem(protocol)
    full_paths = [f"{protocol}://{p}" for p in fs.glob(cloud_path + "*.parquet")]
    return full_paths


def main():
    parser = argparse.ArgumentParser(description="Create a DuckDB database with HNSW and spatial indexes from parquet files.")
    parser.add_argument("parquet_files", nargs='+', help="Paths to parquet files or directories (local or gs:// or s3://)")
    parser.add_argument("output_path", help="Output path for the DuckDB database (local path or gs:// or s3:// path)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index")
    parser.add_argument("--embedding-column", default="embedding", help="Name of the embedding column")
    parser.add_argument("--id-column", default="id", help="Name of the ID column")
    parser.add_argument("--geometry-column", help="Name of the geometry column (auto-detected if not specified, required for database creation)")
    parser.add_argument("--embedding-dim", type=int, help="Dimension of embeddings (auto-detected if not specified)")
    parser.add_argument("--skip-nan-check", action="store_true", help="Skip NaN checking and cleaning")
    
    args = parser.parse_args()
    
    setup_logging()
    
    all_parquet_files = []
    cloud_files = []
    
    for path in args.parquet_files:
        if get_cloud_protocol(path):
            if path.endswith('.parquet'):
                cloud_files.append(path)
            else:
                cloud_parquet_files = list_cloud_parquet_files(path)
                cloud_files.extend(cloud_parquet_files)
                logging.info(f"Found {len(cloud_parquet_files)} parquet files in {path}")
        else:
            path_obj = pathlib.Path(path)
            if path_obj.is_file() and path_obj.suffix == '.parquet':
                all_parquet_files.append(str(path_obj))
            elif path_obj.is_dir():
                parquet_files = list(path_obj.glob("*.parquet"))
                all_parquet_files.extend([str(f) for f in parquet_files])
            else:
                logging.warning(f"Skipping {path} - not a parquet file or directory")
    
    temp_dir = None
    if cloud_files:
        temp_dir = tempfile.mkdtemp(prefix="geovibes_db_")
        logging.info(f"Downloading {len(cloud_files)} files from cloud to temporary directory...")
        local_cloud_files = download_cloud_files(cloud_files, temp_dir)
        all_parquet_files.extend(local_cloud_files)
    
    if not all_parquet_files:
        logging.error("No parquet files found!")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return
    
    logging.info(f"Total {len(all_parquet_files)} parquet files to process")
    
    if get_cloud_protocol(args.output_path):
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
        
        upload_to_cloud(local_temp_file, args.output_path)
        
        os.remove(local_temp_file)
        logging.info(f"Cleaned up temporary file: {local_temp_file}")
    else:
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
    
    if temp_dir:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()