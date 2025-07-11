#!/usr/bin/env python3
"""
Legacy script for processing Google Satellite embeddings.
This functionality has been split into two separate scripts:
- geojson_to_parquet.py: Convert GeoJSON files to GeoParquet format
- ../database.py: Create DuckDB databases from parquet files
"""

import argparse
import logging
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="[DEPRECATED] Process Google Satellite GeoJSON files and create a DuckDB index.")
    parser.add_argument("roi_file", help="Path to the ROI file (e.g., aoi.geojson).")
    parser.add_argument("output_dir", help="Directory to save output files.")
    parser.add_argument("output_db_file", help="Path for the output DuckDB database file.")
    parser.add_argument("--mgrs_reference_file", default="/Users/christopherren/geovibes/geometries/mgrs_tiles.parquet", help="Path to the MGRS grid reference file.")
    parser.add_argument("--gcs_bucket", default="geovibes", help="GCS bucket to use for the embeddings.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of the embedding vectors.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers for processing files.")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.warning("This script is deprecated. Using the new modular approach...")
    
    # Step 1: Convert GeoJSON to Parquet
    cmd1 = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "geojson_to_parquet.py"),
        args.roi_file,
        args.output_dir,
        "--mgrs_reference_file", args.mgrs_reference_file,
        "--gcs_bucket", args.gcs_bucket,
        "--workers", str(args.workers)
    ]
    
    logging.info("Step 1: Converting GeoJSON files to Parquet...")
    result1 = subprocess.run(cmd1)
    if result1.returncode != 0:
        logging.error("Failed to convert GeoJSON files to Parquet")
        sys.exit(1)
    
    # Step 2: Create DuckDB database
    cmd2 = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "database.py"),
        args.output_dir,  # Directory containing parquet files
        args.output_db_file,
        "--metric", args.metric,
        "--embedding-dim", str(args.embedding_dim),
        "--id-column", "tile_id"
    ]
    
    logging.info("Step 2: Creating DuckDB database...")
    result2 = subprocess.run(cmd2)
    if result2.returncode != 0:
        logging.error("Failed to create DuckDB database")
        sys.exit(1)
    
    logging.info("Process completed successfully!")


if __name__ == "__main__":
    main()