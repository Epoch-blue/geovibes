#!/usr/bin/env python3
"""
Transform large parquet file with separate VIT features and WKB polygon geometries
into format compatible with database.py (single embedding column + centroid points).

This script uses DuckDB for efficient processing of large parquet files.
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
import pyproj
from pyproj import CRS, Transformer


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_vit_columns_from_parquet(parquet_file: str) -> list:
    """Get all VIT feature column names from the parquet file."""
    con = duckdb.connect()
    
    # Get column info
    columns_info = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{parquet_file}') LIMIT 1
    """).fetchall()
    
    # Extract VIT columns
    vit_columns = [col[0] for col in columns_info if 'vit' in col[0].lower()]
    con.close()
    
    logging.info(f"Found {len(vit_columns)} VIT feature columns")
    return vit_columns


def determine_utm_zone(geometries_sample):
    """Determine the best UTM zone for a sample of geometries."""
    # Calculate centroids of sample geometries and get average lon/lat
    centroids = []
    for geom_wkb in geometries_sample:
        try:
            geom = wkb.loads(bytes(geom_wkb))
            centroid = geom.centroid
            centroids.append((centroid.x, centroid.y))
        except:
            continue
    
    if not centroids:
        # Default to UTM zone 13N (New Mexico area)
        return "EPSG:32613"
    
    # Calculate average longitude to determine UTM zone
    avg_lon = sum(c[0] for c in centroids) / len(centroids)
    
    # Calculate UTM zone from longitude
    utm_zone = int((avg_lon + 180) / 6) + 1
    
    # For New Mexico, this should be around zone 12-13, assume northern hemisphere
    utm_epsg = f"EPSG:{32600 + utm_zone}"
    
    logging.info(f"Determined UTM zone: {utm_epsg} (avg_lon: {avg_lon:.4f})")
    return utm_epsg


def transform_parquet_with_duckdb(
    input_file: str, 
    output_file: str, 
    limit_rows: Optional[int] = None,
    utm_epsg: Optional[str] = None
):
    """Transform parquet file using DuckDB for efficiency."""
    
    logging.info(f"Starting transformation of {input_file}")
    logging.info(f"Output will be saved to {output_file}")
    
    if limit_rows:
        logging.info(f"Processing only first {limit_rows} rows for testing")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Load necessary extensions
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
        logging.info("DuckDB spatial extension loaded")
    except:
        logging.warning("Could not load DuckDB spatial extension, will use pandas/geopandas")
    
    # Get VIT columns
    vit_columns = get_vit_columns_from_parquet(input_file)
    
    # Build SQL query to read and transform the data
    limit_clause = f"LIMIT {limit_rows}" if limit_rows else ""
    
    # Create VIT array using DuckDB list functions - quote column names that contain dashes
    quoted_vit_columns = [f'"{col}"' for col in vit_columns]
    vit_array_sql = f"[{', '.join(quoted_vit_columns)}]"
    
    logging.info("Reading data with DuckDB...")
    start_time = time.time()
    
    # Read the data
    query = f"""
    SELECT 
        tile_id,
        {vit_array_sql} as embedding,
        geometry
    FROM read_parquet('{input_file}')
    {limit_clause}
    """
    
    result = con.execute(query).fetchall()
    column_names = ['tile_id', 'embedding', 'geometry']
    
    read_time = time.time() - start_time
    logging.info(f"Read {len(result):,} rows in {read_time:.2f} seconds")
    
    con.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(data=result)
    df.columns = column_names
    
    # Process geometries and calculate centroids
    logging.info("Processing geometries and calculating centroids...")
    start_geom_time = time.time()
    
    # Sample geometries to determine UTM zone if not provided
    if utm_epsg is None:
        sample_size = min(100, len(df))
        sample_geometries = df['geometry'].head(sample_size).tolist()
        utm_epsg = determine_utm_zone(sample_geometries)
    
    # Process geometries - use simple iteration like in debug script
    processed_geoms = []
    total_rows = len(df)
    
    logging.info(f"Processing {total_rows} geometries...")
    for idx, row in df.iterrows():
        row_num = int(idx)  # type: ignore
        if row_num % 1000 == 0:
            logging.info(f"Processing row {row_num}/{total_rows}")
        
        try:
            # Convert WKB bytes to shapely geometry
            polygon = wkb.loads(row['geometry'])  # type: ignore
            
            # Transform to UTM for accurate centroid calculation
            temp_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
            temp_gdf_utm = temp_gdf.to_crs(utm_epsg)
            
            # Calculate centroid in UTM
            centroid_utm = temp_gdf_utm.geometry.iloc[0].centroid
            
            # Convert back to WGS84
            temp_centroid_gdf = gpd.GeoDataFrame([1], geometry=[centroid_utm], crs=utm_epsg)
            centroid_wgs84 = temp_centroid_gdf.to_crs("EPSG:4326").geometry.iloc[0]
            
            processed_geoms.append(centroid_wgs84)
            
        except Exception as e:
            logging.warning(f"Error processing geometry at row {row_num}: {e}")
            # Fallback: create a point at 0,0
            processed_geoms.append(Point(0, 0))
    
    # Update DataFrame with centroids (now proper shapely geometry objects)
    df['geometry'] = processed_geoms
    
    geom_time = time.time() - start_geom_time
    logging.info(f"Processed geometries in {geom_time:.2f} seconds")
    
    # Convert to GeoDataFrame for proper parquet saving
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Convert embeddings to float32 for better performance and compatibility
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
    logging.info(f"  UTM zone used: {utm_epsg}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Transform large parquet file for GeoVibes database compatibility")
    
    parser.add_argument("input_file", help="Input parquet file path")
    parser.add_argument("output_file", help="Output parquet file path")
    parser.add_argument("--limit-rows", type=int, help="Limit processing to N rows (for testing)")
    parser.add_argument("--utm-epsg", help="UTM EPSG code (e.g., EPSG:32613), auto-detected if not specified")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logging.error(f"Input file does not exist: {args.input_file}")
        return 1
    
    try:
        transform_parquet_with_duckdb(
            args.input_file, 
            args.output_file, 
            limit_rows=args.limit_rows,
            utm_epsg=args.utm_epsg
        )
        return 0
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 