#!/usr/bin/env python3
"""
Process-based, memory-efficient transformation script for large parquet files.
Each worker process reads, transforms, and writes its own slice of data.
Outputs partitioned parquet files that can be queried as a single dataset.

Earth Genome parquet files are stored with each element of the embedding as a separate column.
They also store full polygon geometries, but we only need the centroids.
This script:
- Converts the embedding columns into a single array column
- Calculates polygon centroids using vectorized UTM projections for accuracy
- Groups geometries by UTM zone for efficient batch projection
- Projects centroids back to EPSG:4326
- Currently only supports ViT embeddings (hardcoded)
"""

import argparse
import logging
import time
import os
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb


def setup_logging() -> None:
    """Configure logging with timestamp and level information."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_total_rows(parquet_file: str) -> int:
    """
    Get total number of rows in the parquet file.
    
    Args:
        parquet_file: Path to the input parquet file
        
    Returns:
        Total number of rows in the file
    """
    con = duckdb.connect()
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
    con.close()
    return result[0] if result else 0


def get_vit_columns(parquet_file: str) -> List[str]:
    """
    Extract all VIT feature column names from the parquet file.
    
    Args:
        parquet_file: Path to the input parquet file
        
    Returns:
        List of column names containing 'vit' (case-insensitive)
    """
    con = duckdb.connect()
    columns_info = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{parquet_file}') LIMIT 1
    """).fetchall()
    
    # Check if geometry column exists and its type
    geometry_type = None
    for col_name, col_type, *_ in columns_info:
        if col_name.lower() == 'geometry':
            geometry_type = col_type
            logging.info(f"Found geometry column with type: {col_type}")
            break
    
    con.close()
    
    vit_columns = [col[0] for col in columns_info if 'vit' in col[0].lower()]
    return vit_columns


def add_utm_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Vectorized centroid calculation using appropriate UTM projections.
    
    1. Derive UTM zone per feature using vectorized bounds calculations
    2. Group by UTM zone and project each group to its UTM CRS
    3. Calculate centroids (vectorized) and project back to EPSG:4326
    4. Concatenate results maintaining original order
    
    Args:
        gdf: GeoDataFrame with polygon geometries in EPSG:4326
        
    Returns:
        GeoDataFrame with centroid geometries in EPSG:4326
    """
    bounds = np.vstack(gdf.geometry.bounds.values)
    ctr_lon = (bounds[:, 0] + bounds[:, 2]) / 2
    ctr_lat = (bounds[:, 1] + bounds[:, 3]) / 2
    zones = ((ctr_lon + 180) // 6 + 1).astype(int)
    north = ctr_lat >= 0
    epsg = np.where(north, 32600 + zones, 32700 + zones)
    gdf = gdf.copy()
    gdf["utm_epsg"] = epsg

    out_frames = []
    for e, grp in gdf.groupby("utm_epsg"):
        g = grp.to_crs(f"EPSG:{int(e)}")
        cent = g.centroid
        cent = gpd.GeoSeries(cent, crs=g.crs).to_crs("EPSG:4326")
        grp = grp.drop(columns=["geometry", "utm_epsg"]).set_geometry(cent)
        out_frames.append(grp)

    return gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs="EPSG:4326")


def compute_chunks(total_rows: int, batch_size: int) -> List[Tuple[int, int]]:
    """
    Pre-compute all chunk offsets and sizes for parallel processing.
    
    Args:
        total_rows: Total number of rows to process
        batch_size: Number of rows per chunk
        
    Returns:
        List of (offset, size) tuples for each chunk
    """
    chunks = []
    for offset in range(0, total_rows, batch_size):
        size = min(batch_size, total_rows - offset)
        chunks.append((offset, size))
    return chunks


def worker_process_chunk(
    parquet_file: str,
    vit_columns: List[str],
    chunk_info: Tuple[int, int],
    output_dir: str
) -> Tuple[int, int, float]:
    """
    Process a single chunk of data in a separate process.
    
    Reads a slice of the parquet file, converts polygon geometries to centroids
    using vectorized UTM projections for accuracy, transforms embeddings to float32,
    and writes to a partitioned output file.
    
    Args:
        parquet_file: Path to the input parquet file
        vit_columns: List of VIT feature column names
        chunk_info: Tuple of (offset, size) for this chunk
        output_dir: Directory to write partitioned output files
        
    Returns:
        Tuple of (offset, rows_processed, time_taken)
    """
    offset, size = chunk_info
    start_time = time.time()
    
    try:
        con = duckdb.connect()
        
        # Try to load spatial extension, but don't fail if it's not available
        try:
            con.execute("INSTALL spatial")
            con.execute("LOAD spatial")
        except Exception as e:
            logging.debug(f"Could not load spatial extension: {e}")
        
        quoted_vit_columns = [f'"{col}"' for col in vit_columns]
        vit_array_sql = f"[{', '.join(quoted_vit_columns)}]"
        
        # First try: Assume geometry is already a GEOMETRY type and needs ST_AsWKB
        query_with_spatial = f"""
        SELECT 
            tile_id,
            {vit_array_sql} as embedding,
            ST_AsWKB(geometry) as geometry
        FROM read_parquet('{parquet_file}')
        LIMIT {size} OFFSET {offset}
        """
        
        # Second try: Assume geometry is already WKB bytes
        query_direct = f"""
        SELECT 
            tile_id,
            {vit_array_sql} as embedding,
            geometry
        FROM read_parquet('{parquet_file}')
        LIMIT {size} OFFSET {offset}
        """
        
        # Try with spatial extension first
        try:
            result = con.execute(query_with_spatial).fetchall()
            logging.debug("Successfully used ST_AsWKB for geometry column")
        except Exception as e:
            logging.debug(f"ST_AsWKB failed: {e}, trying direct geometry access")
            result = con.execute(query_direct).fetchall()
        
        con.close()
        
        df = pd.DataFrame(data=result)
        df.columns = ['tile_id', 'embedding', 'geometry']
        
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Convert WKB geometries to shapely objects
        df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if x is not None else None)
        
        # Create GeoDataFrame and calculate UTM centroids
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        gdf = add_utm_centroid(gdf)
        
        output_file = os.path.join(output_dir, f"part_{offset:012d}.parquet")
        gdf.to_parquet(output_file, index=False)
        
        elapsed = time.time() - start_time
        rows_processed = len(df)
        
        return offset, rows_processed, elapsed
        
    except Exception as e:
        logging.error(f"Worker error processing chunk at offset {offset}: {e}")
        return offset, 0, time.time() - start_time


def merge_partitioned_output(output_dir: str, final_output: str) -> None:
    """
    Merge all partition files into a single output using DuckDB.
    
    This streams the data efficiently without loading everything into memory.
    
    Args:
        output_dir: Directory containing partitioned parquet files
        final_output: Path for the merged output file
    """
    logging.info(f"Merging partitioned files from {output_dir} to {final_output}")
    
    con = duckdb.connect()
    pattern = os.path.join(output_dir, "*.parquet")
    
    try:
        count_result = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{pattern}')
        """).fetchone()
        total_rows = count_result[0] if count_result else 0
        
        logging.info(f"Merging {total_rows:,} rows from partitioned files")
        
        con.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{pattern}')
                ORDER BY tile_id
            )
            TO '{final_output}'
            (FORMAT PARQUET)
        """)
        
        logging.info(f"Successfully created merged file: {final_output}")
        
    finally:
        con.close()


def transform_parquet_processes(
    input_file: str,
    output_dir: str,
    batch_size: int = 1500000,
    limit_rows: Optional[int] = None,
    max_workers: int = 4,
    single_output: Optional[str] = None
) -> str:
    """
    Transform parquet file using process-based parallel execution.
    
    Each process handles a complete slice independently, reading from the input,
    transforming the data, and writing to a partitioned output file.
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory for partitioned output files
        batch_size: Number of rows per batch (default: 1.5M)
        limit_rows: Optional limit on total rows to process
        max_workers: Maximum number of worker processes (default: 4)
        single_output: Optional path for merged single output file
        
    Returns:
        Path to the output directory
    """
    logging.info(f"Starting PROCESS-BASED transformation of {input_file}")
    logging.info(f"Batch size: {batch_size:,} rows")
    logging.info(f"Max workers: {max_workers}")
    logging.info(f"Output directory: {output_dir}")
    if single_output:
        logging.info(f"Will merge to single file: {single_output}")
    
    start_time = time.time()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_rows = get_total_rows(input_file)
    if limit_rows and limit_rows < total_rows:
        total_rows = limit_rows
        logging.info(f"Limited to first {limit_rows:,} rows")
    
    vit_columns = get_vit_columns(input_file)
    chunks = compute_chunks(total_rows, batch_size)
    
    logging.info(f"Dataset info:")
    logging.info(f"  Total rows: {total_rows:,}")
    logging.info(f"  VIT features: {len(vit_columns)}")
    logging.info(f"  Batch size: {batch_size:,}")
    logging.info(f"  Total chunks: {len(chunks)}")
    
    completed_rows = 0
    completed_chunks = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                worker_process_chunk,
                input_file,
                vit_columns,
                chunk,
                output_dir
            ): chunk
            for chunk in chunks
        }
        
        for future in as_completed(future_to_chunk):
            offset, rows_processed, elapsed = future.result()
            
            if rows_processed > 0:
                completed_rows += rows_processed
                rate = rows_processed / elapsed if elapsed > 0 else 0
                
                logging.info(
                    f"Chunk at offset {offset:,}: Processed {rows_processed:,} rows "
                    f"in {elapsed:.2f}s ({rate:.0f} rows/sec)"
                )
            else:
                logging.error(f"Chunk at offset {offset:,}: Failed")
            
            completed_chunks += 1
            
            progress_pct = (completed_rows / total_rows) * 100 if total_rows > 0 else 0
            overall_elapsed = time.time() - start_time
            overall_rate = completed_rows / overall_elapsed if overall_elapsed > 0 else 0
            eta = (total_rows - completed_rows) / overall_rate if overall_rate > 0 else 0
            
            logging.info(
                f"PROGRESS: {completed_rows:,}/{total_rows:,} ({progress_pct:.1f}%) | "
                f"Rate: {overall_rate:.0f} rows/sec | ETA: {eta/60:.1f} min | "
                f"Chunks: {completed_chunks}/{len(chunks)}"
            )
    
    if single_output:
        merge_start = time.time()
        merge_partitioned_output(output_dir, single_output)
        merge_time = time.time() - merge_start
        logging.info(f"Merge completed in {merge_time:.2f} seconds")
    
    total_time = time.time() - start_time
    final_rate = total_rows / total_time if total_time > 0 else 0
    
    logging.info(f"TRANSFORMATION COMPLETE!")
    logging.info(f"  Input: {input_file}")
    logging.info(f"  Output: {output_dir}")
    if single_output:
        logging.info(f"  Single file: {single_output}")
    logging.info(f"  Rows processed: {total_rows:,}")
    logging.info(f"  Total time: {total_time/60:.2f} minutes")
    logging.info(f"  Average rate: {final_rate:.0f} rows/sec")
    logging.info(f"  Workers used: {max_workers}")
    
    return output_dir


def main() -> int:
    """
    Command-line interface for the parquet transformation script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Transform large parquet files using process-based parallelism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - outputs partitioned directory
  %(prog)s input.parquet output_dir/
  
  # With custom batch size and workers
  %(prog)s input.parquet output_dir/ --batch-size 2000000 --max-workers 8
  
  # Create single merged file
  %(prog)s input.parquet output_dir/ --single-output merged.parquet
        """
    )
    
    parser.add_argument("input_file", help="Input parquet file path")
    parser.add_argument("output_dir", help="Output directory for partitioned files")
    parser.add_argument("--batch-size", type=int, default=1500000, 
                        help="Rows per batch (default: 1.5M)")
    parser.add_argument("--limit-rows", type=int, 
                        help="Limit processing to N rows (for testing)")
    parser.add_argument("--max-workers", type=int, default=4, 
                        help="Maximum worker processes (default: 4)")
    parser.add_argument("--single-output", type=str,
                        help="Path for optional single merged output file")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not os.path.exists(args.input_file):
        logging.error(f"Input file does not exist: {args.input_file}")
        return 1
    
    output_dir = os.path.normpath(args.output_dir)
    
    try:
        transform_parquet_processes(
            args.input_file,
            output_dir,
            batch_size=args.batch_size,
            limit_rows=args.limit_rows,
            max_workers=args.max_workers,
            single_output=args.single_output
        )
        return 0
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())