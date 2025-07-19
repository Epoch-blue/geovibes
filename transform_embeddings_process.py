#!/usr/bin/env python3
"""
Process-based, memory-efficient transformation script for large parquet files.
Each worker process reads, transforms, and writes its own slice of data.
Outputs partitioned parquet files that can be queried as a single dataset.
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
from shapely.geometry import Point


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_total_rows(parquet_file: str) -> int:
    """Get total number of rows in the parquet file."""
    con = duckdb.connect()
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
    con.close()
    return result[0] if result else 0


def get_vit_columns(parquet_file: str) -> List[str]:
    """Get all VIT feature column names from the parquet file."""
    con = duckdb.connect()
    columns_info = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{parquet_file}') LIMIT 1
    """).fetchall()
    con.close()
    
    vit_columns = [col[0] for col in columns_info if 'vit' in col[0].lower()]
    return vit_columns


def process_geometry_batch(geometry_data: List[bytes]) -> List[Point]:
    """Process a batch of WKB geometries efficiently."""
    results = []
    error_count = 0
    
    for geom_wkb in geometry_data:
        try:
            polygon = wkb.loads(geom_wkb)
            results.append(polygon.centroid)
        except Exception:
            error_count += 1
            results.append(Point(0, 0))  # Fallback
    
    if error_count > 0:
        logging.debug(f"Geometry processing errors in batch: {error_count}/{len(geometry_data)}")
    
    return results


def compute_chunks(total_rows: int, batch_size: int) -> List[Tuple[int, int]]:
    """Pre-compute all chunk offsets and sizes."""
    chunks = []
    for offset in range(0, total_rows, batch_size):
        size = min(batch_size, total_rows - offset)
        chunks.append((offset, size))
    return chunks


def worker_process_chunk(
    parquet_file: str,
    vit_columns: List[str],
    chunk_info: Tuple[int, int],
    output_dir: str,
    use_spatial_extension: bool = False
) -> Tuple[int, int, float]:
    """
    Worker function that processes a single chunk of data.
    Runs in a separate process to avoid GIL limitations.
    
    Note: Handles both regular parquet (WKB bytes) and GeoParquet files.
    When use_spatial_extension=True, uses DuckDB spatial for centroid computation.
    
    Returns: (offset, rows_processed, time_taken)
    """
    offset, size = chunk_info
    start_time = time.time()
    
    try:
        # Each worker creates its own DuckDB connection
        con = duckdb.connect()
        
        # Load spatial extension if requested (optional optimization)
        if use_spatial_extension:
            try:
                con.execute("INSTALL spatial")
                con.execute("LOAD spatial")
                use_spatial_sql = True
            except:
                use_spatial_sql = False
        else:
            use_spatial_sql = False
        
        # Create the VIT array SQL
        quoted_vit_columns = [f'"{col}"' for col in vit_columns]
        vit_array_sql = f"[{', '.join(quoted_vit_columns)}]"
        
        # Build query with optional spatial centroid computation
        if use_spatial_sql:
            # For GeoParquet files with WKB geometry columns
            query = f"""
            SELECT 
                tile_id,
                {vit_array_sql} as embedding,
                ST_AsWKB(ST_Centroid(ST_GeomFromWKB(geometry))) as geometry
            FROM read_parquet('{parquet_file}')
            LIMIT {size} OFFSET {offset}
            """
        else:
            query = f"""
            SELECT 
                tile_id,
                {vit_array_sql} as embedding,
                geometry
            FROM read_parquet('{parquet_file}')
            LIMIT {size} OFFSET {offset}
            """
        
        # Execute query and fetch results
        result = con.execute(query).fetchall()
        con.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(data=result)
        df.columns = ['tile_id', 'embedding', 'geometry']
        
        # Process geometries if not using spatial SQL
        if not use_spatial_sql:
            # Check if geometry is WKB bytes or already geometry objects
            first_geom = df['geometry'].iloc[0] if len(df) > 0 else None

            geometry_data = df['geometry'].tolist()
            processed_geoms = process_geometry_batch(geometry_data)
            df['geometry'] = processed_geoms

        
        # Convert embeddings to float32 numpy arrays
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Write to partitioned output
        output_file = os.path.join(output_dir, f"part_{offset:012d}.parquet")
        gdf.to_parquet(output_file, index=False)
        
        elapsed = time.time() - start_time
        rows_processed = len(df)
        
        return offset, rows_processed, elapsed
        
    except Exception as e:
        logging.error(f"Worker error processing chunk at offset {offset}: {e}")
        return offset, 0, time.time() - start_time


def merge_partitioned_output(output_dir: str, final_output: str):
    """
    Use DuckDB to efficiently merge all partition files into a single output.
    This streams the data and is much more efficient than pd.concat.
    """
    logging.info(f"Merging partitioned files from {output_dir} to {final_output}")
    
    con = duckdb.connect()
    
    # Pattern to read all parquet files in the directory
    pattern = os.path.join(output_dir, "*.parquet")
    
    try:
        # Count total rows for verification
        count_result = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{pattern}')
        """).fetchone()
        total_rows = count_result[0] if count_result else 0
        
        logging.info(f"Merging {total_rows:,} rows from partitioned files")
        
        # Execute COPY command to stream all partitions into single file
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
    single_output: Optional[str] = None,
    use_spatial_extension: bool = False
):
    """
    Transform parquet file using process-based parallel execution.
    Each process handles a complete slice independently.
    """
    
    logging.info(f"Starting PROCESS-BASED transformation of {input_file}")
    logging.info(f"Batch size: {batch_size:,} rows")
    logging.info(f"Max workers: {max_workers}")
    logging.info(f"Output directory: {output_dir}")
    if single_output:
        logging.info(f"Will merge to single file: {single_output}")
    
    start_time = time.time()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get dataset info
    total_rows = get_total_rows(input_file)
    if limit_rows and limit_rows < total_rows:
        total_rows = limit_rows
        logging.info(f"Limited to first {limit_rows:,} rows")
    
    vit_columns = get_vit_columns(input_file)
    
    # Pre-compute all chunks
    chunks = compute_chunks(total_rows, batch_size)
    
    logging.info(f"Dataset info:")
    logging.info(f"  Total rows: {total_rows:,}")
    logging.info(f"  VIT features: {len(vit_columns)}")
    logging.info(f"  Total chunks: {len(chunks)}")
    
    # Process chunks in parallel using processes
    completed_rows = 0
    completed_chunks = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks to the process pool
        future_to_chunk = {
            executor.submit(
                worker_process_chunk,
                input_file,
                vit_columns,
                chunk,
                output_dir,
                use_spatial_extension
            ): chunk
            for chunk in chunks
        }
        
        # Process results as they complete
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
            
            # Progress reporting
            progress_pct = (completed_rows / total_rows) * 100 if total_rows > 0 else 0
            overall_elapsed = time.time() - start_time
            overall_rate = completed_rows / overall_elapsed if overall_elapsed > 0 else 0
            eta = (total_rows - completed_rows) / overall_rate if overall_rate > 0 else 0
            
            logging.info(
                f"PROGRESS: {completed_rows:,}/{total_rows:,} ({progress_pct:.1f}%) | "
                f"Rate: {overall_rate:.0f} rows/sec | ETA: {eta/60:.1f} min | "
                f"Chunks: {completed_chunks}/{len(chunks)}"
            )
    
    # Optionally merge to single file
    if single_output:
        merge_start = time.time()
        merge_partitioned_output(output_dir, single_output)
        merge_time = time.time() - merge_start
        logging.info(f"Merge completed in {merge_time:.2f} seconds")
    
    # Final summary
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


def main():
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
  
  # Use DuckDB spatial extension for geometry processing
  %(prog)s input.parquet output_dir/ --use-spatial
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
    parser.add_argument("--use-spatial", action="store_true",
                        help="Use DuckDB spatial extension for geometry processing")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logging.error(f"Input file does not exist: {args.input_file}")
        return 1
    
    # Ensure output directory doesn't have trailing slash issues
    output_dir = os.path.normpath(args.output_dir)
    
    try:
        transform_parquet_processes(
            args.input_file,
            output_dir,
            batch_size=args.batch_size,
            limit_rows=args.limit_rows,
            max_workers=args.max_workers,
            single_output=args.single_output,
            use_spatial_extension=args.use_spatial
        )
        return 0
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())