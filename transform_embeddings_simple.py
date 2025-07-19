#!/usr/bin/env python3
"""
Simplified transformation script that converts the large parquet file 
to the format needed by database.py. Uses proven working approach.
"""

import argparse
import logging
import time
import os
from pathlib import Path
from typing import Optional

import duckdb
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb
from shapely.geometry import Point


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def transform_parquet_simple(
    input_file: str, 
    output_file: str, 
    limit_rows: Optional[int] = None
):
    """Transform parquet file using the proven working approach from debug script."""
    
    logging.info(f"Starting transformation of {input_file}")
    logging.info(f"Output will be saved to {output_file}")
    
    if limit_rows:
        logging.info(f"Processing only first {limit_rows} rows for testing")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Get VIT columns
    logging.info("Getting VIT column names...")
    columns_info = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{input_file}') LIMIT 1
    """).fetchall()
    
    vit_columns = [col[0] for col in columns_info if 'vit' in col[0].lower()]
    quoted_vit_columns = [f'"{col}"' for col in vit_columns]
    vit_array_sql = f"[{', '.join(quoted_vit_columns)}]"
    
    logging.info(f"Found {len(vit_columns)} VIT feature columns")
    
    # Build SQL query
    limit_clause = f"LIMIT {limit_rows}" if limit_rows else ""
    
    query = f"""
    SELECT 
        tile_id,
        {vit_array_sql} as embedding,
        geometry
    FROM read_parquet('{input_file}')
    {limit_clause}
    """
    
    logging.info("Reading data with DuckDB...")
    start_time = time.time()
    
    result = con.execute(query).fetchall()
    con.close()
    
    read_time = time.time() - start_time
    logging.info(f"Read {len(result):,} rows in {read_time:.2f} seconds")
    
    # Convert to DataFrame
    df = pd.DataFrame(data=result)
    df.columns = ['tile_id', 'embedding', 'geometry']
    
    # Process geometries using the working approach
    logging.info("Processing geometries and calculating centroids...")
    start_geom_time = time.time()
    
    processed_geoms = []
    total_rows = len(df)
    error_count = 0
    
    # Progress tracking
    progress_interval = max(1, total_rows // 100)  # Show progress every 1%
    if total_rows > 10000:
        progress_interval = max(progress_interval, 10000)  # But at least every 10k rows
    
    logging.info(f"Will show progress every {progress_interval:,} rows")
    
    for i, row in df.iterrows():
        row_num = int(i) if isinstance(i, int) else len(processed_geoms)
        
        # Enhanced progress tracking
        if row_num % progress_interval == 0 and row_num > 0:
            elapsed = time.time() - start_geom_time
            progress_pct = (row_num / total_rows) * 100
            rate = row_num / elapsed if elapsed > 0 else 0
            eta = (total_rows - row_num) / rate if rate > 0 else 0
            
            logging.info(
                f"Progress: {row_num:,}/{total_rows:,} ({progress_pct:.1f}%) | "
                f"Rate: {rate:.0f} rows/sec | "
                f"ETA: {eta/60:.1f} min | "
                f"Errors: {error_count}"
            )
        
        try:
            # Parse WKB and calculate centroid (simple approach that works)
            polygon = wkb.loads(row['geometry'])  # type: ignore
            centroid = polygon.centroid
            processed_geoms.append(centroid)
            
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only log first 10 errors to avoid spam
                logging.warning(f"Error processing geometry at row {row_num}: {e}")
            elif error_count == 11:
                logging.warning(f"More geometry errors occurring... will only log summary")
            # Fallback: create a point at 0,0
            processed_geoms.append(Point(0, 0))
    
    # Update DataFrame with centroids
    df['geometry'] = processed_geoms
    
    geom_time = time.time() - start_geom_time
    final_rate = total_rows / geom_time if geom_time > 0 else 0
    
    logging.info(f"Geometry processing complete!")
    logging.info(f"  Time: {geom_time:.2f} seconds")
    logging.info(f"  Rate: {final_rate:.0f} rows/sec")
    logging.info(f"  Success: {total_rows - error_count:,}/{total_rows:,}")
    logging.info(f"  Errors: {error_count:,} ({error_count/total_rows*100:.3f}%)")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Convert embeddings to float32 for better performance
    logging.info("Converting embeddings to float32...")
    gdf['embedding'] = gdf['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    
    # Save to parquet
    logging.info(f"Saving transformed data to {output_file}...")
    start_save_time = time.time()
    
    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf.to_parquet(output_file, index=False)
    
    save_time = time.time() - start_save_time
    logging.info(f"Saved {len(gdf):,} rows to {output_file} in {save_time:.2f} seconds")
    
    # Summary
    total_time = time.time() - start_time
    logging.info(f"TRANSFORMATION COMPLETE!")
    logging.info(f"  Input: {input_file}")
    logging.info(f"  Output: {output_file}")
    logging.info(f"  Rows: {len(gdf):,}")
    logging.info(f"  Embedding dimension: {len(gdf['embedding'].iloc[0])}")
    logging.info(f"  Total time: {total_time:.2f} seconds")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Transform large parquet file for GeoVibes database compatibility (simplified version)")
    
    parser.add_argument("input_file", help="Input parquet file path")
    parser.add_argument("output_file", help="Output parquet file path")
    parser.add_argument("--limit-rows", type=int, help="Limit processing to N rows (for testing)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logging.error(f"Input file does not exist: {args.input_file}")
        return 1
    
    try:
        transform_parquet_simple(
            args.input_file, 
            args.output_file, 
            limit_rows=args.limit_rows
        )
        return 0
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 