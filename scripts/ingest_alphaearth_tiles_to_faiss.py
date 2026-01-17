#!/usr/bin/env python3
"""
Ingest AlphaEarth embedding tiles (GeoJSON) into FAISS database.
This script converts GeoJSON tiles to Parquet format and builds the FAISS index.
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
from typing import List, Optional, Tuple
from tempfile import NamedTemporaryFile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
import pyarrow as pa
import pyarrow.parquet as pq
import time

try:
    import gcsfs  # type: ignore
except Exception:
    gcsfs = None  # optional; required only for --gcs-prefix

def process_single_tile(args: Tuple[str, bool, Optional[List[str]]]) -> Optional[pd.DataFrame]:
    """
    Process a single GeoJSON tile and return DataFrame.
    
    Args:
        args: Tuple of (geojson_file, is_gcs, embedding_bands)
    
    Returns:
        DataFrame with processed features or None if failed
    """
    geojson_file, is_gcs, embedding_bands = args
    start_time = time.time()
    
    try:
        # Read GeoJSON (local or streamed from GCS via temp file)
        if is_gcs:
            download_start = time.time()
            # Create GCS filesystem with optimized settings
            fs = gcsfs.GCSFileSystem(
                cache_timeout=300,  # Cache for 5 minutes
                default_block_size=2**20,  # 1MB blocks for faster reads
                default_fill_cache=True
            )
            with fs.open(geojson_file, 'rb') as fsrc, NamedTemporaryFile(suffix='.geojson') as tmp:
                data = fsrc.read()
                tmp.write(data)
                tmp.flush()
                download_time = time.time() - download_start
                
                parse_start = time.time()
                gdf = gpd.read_file(tmp.name)
                parse_time = time.time() - parse_start
                
                file_size_mb = len(data) / 1024 / 1024
                download_speed = file_size_mb / download_time if download_time > 0 else 0
                
                print(f"    📥 Downloaded {file_size_mb:.1f}MB in {download_time:.1f}s ({download_speed:.1f}MB/s)")
                print(f"    📊 Parsed GeoJSON in {parse_time:.1f}s")
        else:
            gdf = gpd.read_file(geojson_file)
        
        if gdf.empty:
            return None
        
        # Auto-detect embedding bands if not provided
        if embedding_bands is None:
            embedding_bands = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
        
        if not embedding_bands:
            return None
        
        # Process features
        tile_features = []
        display_name = Path(geojson_file).name if not is_gcs else geojson_file.split('/')[-1]
        
        for idx, row in gdf.iterrows():
            # Extract coordinates
            if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                lon, lat = row.geometry.x, row.geometry.y
            else:
                # For other geometry types, get centroid
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            
            # Extract embedding values
            embedding_values = [row[band] for band in embedding_bands if band in row]
            
            if len(embedding_values) != len(embedding_bands):
                continue
            
            # Create feature record
            feature = {
                'lon': lon,
                'lat': lat,
                'tile_id': Path(display_name).stem
            }
            
            # Add embedding values as individual columns
            for j, band in enumerate(embedding_bands):
                feature[band] = embedding_values[j]
                
                # Also create combined embedding column for FAISS compatibility
                # Convert to numpy array for proper DuckDB storage
                import numpy as np
                feature['embedding'] = np.array(embedding_values, dtype=np.float32)
            
            # Add any other properties
            for col in gdf.columns:
                if col not in ['geometry'] + embedding_bands:
                    feature[col] = row[col]
            
            tile_features.append(feature)
        
        if not tile_features:
            return None
        
        total_time = time.time() - start_time
        print(f"    ⏱️  Total processing time: {total_time:.1f}s")
        
        return pd.DataFrame(tile_features)
    
    except Exception as e:
        print(f"  ❌ Error processing {geojson_file}: {e}")
        return None

def convert_geojson_tiles_to_parquet(
    tiles_dir: Optional[str],
    output_parquet: str,
    embedding_bands: Optional[List[str]] = None,
    gcs_prefix: Optional[str] = None,
    max_workers: Optional[int] = None
) -> None:
    """
    Convert GeoJSON tiles to Parquet format for FAISS ingestion.
    Processes tiles in parallel for better performance.
    
    Args:
        tiles_dir: Directory containing GeoJSON tile files (local)
        output_parquet: Output Parquet file path
        embedding_bands: List of embedding band names to include
        gcs_prefix: GCS URI prefix (gs://bucket/prefix) to read tiles directly
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    
    # Discover files either locally or on GCS
    geojson_files: List[str]
    is_gcs = False
    if gcs_prefix:
        if gcsfs is None:
            raise RuntimeError("gcsfs is required for --gcs-prefix. Install with: pip install gcsfs")
        fs = gcsfs.GCSFileSystem()
        prefix = gcs_prefix.rstrip('/')
        geojson_files = fs.glob(f"{prefix}/**/*.geojson") or fs.glob(f"{prefix}/*.geojson")
        if not geojson_files:
            raise ValueError(f"No GeoJSON files found under {gcs_prefix}")
        is_gcs = True
        print(f"🔍 Reading GeoJSON tiles directly from GCS: {gcs_prefix}")
    else:
        assert tiles_dir is not None
        print(f"🔍 Looking for GeoJSON tiles in: {tiles_dir}")
        geojson_files = glob.glob(os.path.join(tiles_dir, "*.geojson"))
        if not geojson_files:
            raise ValueError(f"No GeoJSON files found in {tiles_dir}")
    
    print(f"📦 Found {len(geojson_files)} GeoJSON tiles")
    
    # Set up parallel processing
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(geojson_files))
    
    print(f"🚀 Processing with {max_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    tile_args = [(file, is_gcs, embedding_bands) for file in geojson_files]
    
    # Process tiles in parallel and batch write to Parquet
    processed_tiles = 0
    total_features = 0
    parquet_writer = None
    schema = None
    batch_size = 5  # Write every 5 tiles to reduce I/O
    batch_tables = []
    
    # Use a queue to handle results from parallel processing
    result_queue = queue.Queue()
    
    def process_tiles():
        """Process tiles in parallel and put results in queue"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_tile, args): args[0] for args in tile_args}
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                geojson_file = future_to_file[future]
                display_name = Path(geojson_file).name if not is_gcs else geojson_file.split('/')[-1]
                
                try:
                    tile_df = future.result()
                    result_queue.put((display_name, tile_df, None))
                except Exception as e:
                    result_queue.put((display_name, None, str(e)))
    
    # Start parallel processing in background thread
    process_thread = threading.Thread(target=process_tiles)
    process_thread.start()
    
    def write_batch():
        """Write accumulated tables to Parquet"""
        nonlocal parquet_writer, schema, total_features
        
        if not batch_tables:
            return
        
        try:
            # Combine all tables in batch
            combined_table = pa.concat_tables(batch_tables)
            print(f"  📦 Writing batch of {len(batch_tables)} tables ({len(combined_table)} total rows)")
            
            # Initialize Parquet writer with schema from first table
            if parquet_writer is None:
                schema = combined_table.schema
                parquet_writer = pq.ParquetWriter(output_parquet, schema)
                print(f"💾 Creating streaming Parquet file: {output_parquet}")
            
            # Write combined table to Parquet
            parquet_writer.write_table(combined_table)
            total_features += len(combined_table)
            
            # Check file size
            if os.path.exists(output_parquet):
                file_size = os.path.getsize(output_parquet)
                print(f"  ✅ Batch written: {len(combined_table)} features (total: {total_features}, file: {file_size/1024/1024:.1f}MB)")
            
            # Clear batch
            batch_tables.clear()
            
        except Exception as e:
            print(f"  ❌ Error writing batch to Parquet: {e}")
            batch_tables.clear()
    
    # Stream results to Parquet file with batching
    while processed_tiles < len(geojson_files):
        try:
            display_name, tile_df, error = result_queue.get(timeout=30)
            
            if error:
                print(f"  ❌ Error processing {display_name}: {error}")
            elif tile_df is not None and not tile_df.empty:
                print(f"  🔍 Processing {display_name}: DataFrame shape {tile_df.shape}")
                
                # Auto-detect embedding bands from first successful tile
                if embedding_bands is None:
                    embedding_bands = [col for col in tile_df.columns if col.startswith('A') and col[1:].isdigit()]
                    print(f"  🎯 Auto-detected {len(embedding_bands)} embedding bands: {embedding_bands[:5]}...")
                
                # Convert to PyArrow Table
                try:
                    table = pa.Table.from_pandas(tile_df)
                    batch_tables.append(table)
                    print(f"  📊 Added to batch: {len(table)} rows (batch size: {len(batch_tables)})")
                except Exception as e:
                    print(f"  ❌ Error creating PyArrow table: {e}")
                    continue
                
                # Write batch when it reaches batch_size
                if len(batch_tables) >= batch_size:
                    write_batch()
            else:
                print(f"  ⚠️  {display_name}: No valid features, skipping")
            
            processed_tiles += 1
            
        except queue.Empty:
            print("  ⏳ Waiting for more results...")
            continue
    
    # Write any remaining tables in the batch
    if batch_tables:
        write_batch()
    
    # Close Parquet writer
    if parquet_writer is not None:
        parquet_writer.close()
    
    # Wait for processing thread to complete
    process_thread.join()
    
    if total_features == 0:
        raise ValueError("No valid features found in any tiles")
    
    print(f"✅ Successfully converted {total_features} features from {processed_tiles} tiles to Parquet")
    print(f"📋 Embedding dimensions: {len(embedding_bands) if embedding_bands else 0}")
    print(f"📁 Output file: {output_parquet}")

def ingest_to_faiss_db(
    parquet_file: str,
    db_dir: str,
    embedding_bands: Optional[List[str]] = None
) -> None:
    """
    Ingest Parquet file into FAISS database.
    
    Args:
        parquet_file: Path to Parquet file
        db_dir: Directory for FAISS database
        embedding_bands: List of embedding band names
    """
    
    print("🔍 Ingesting Parquet to FAISS database...")
    
    # Import FAISS database utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from geovibes.database.faiss_db import ingest_parquet_to_duckdb, create_faiss_index
    
    # Create database directory
    os.makedirs(db_dir, exist_ok=True)
    
    # Set file paths (not directories)
    db_path = os.path.join(db_dir, "metadata.db")
    index_path = os.path.join(db_dir, "faiss.index")
    
    # Ingest to DuckDB
    print("📊 Ingesting to DuckDB...")
    parquet_files = [parquet_file]
    dtype = "FLOAT"  # Use 'FLOAT' not 'float32' for FAISS compatibility
    embedding_col = "embedding"
    
    # Capture returned embedding dimension
    embedding_dim = ingest_parquet_to_duckdb(parquet_files, db_path, dtype, embedding_col)
    
    # Create FAISS index with repo's signature
    print("🔍 Creating FAISS index...")
    create_faiss_index(
        db_path=db_path,
        index_path=index_path,
        embedding_dim=embedding_dim,
        dtype=dtype,
        nlist=4096,
        m=64,
        nbits=8,
        batch_size=1_000_000
    )
    
    print("✅ FAISS database created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Ingest AlphaEarth embedding tiles to FAISS database")
    
    parser.add_argument("--tiles-dir", help="Directory containing GeoJSON tile files (local)")
    parser.add_argument("--gcs-prefix", help="GCS prefix containing GeoJSON tiles, e.g. gs://bucket/path")
    parser.add_argument("--output-parquet", help="Output Parquet file path (default: tiles_dir/embeddings.parquet)")
    parser.add_argument("--db-dir", help="FAISS database directory (default: tiles_dir/../faiss_db)")
    parser.add_argument("--embedding-bands", nargs="+", help="List of embedding band names (auto-detected if not provided)")
    parser.add_argument("--max-workers", type=int, help="Maximum number of parallel workers (default: CPU count)")
    parser.add_argument("--download-first", action="store_true", help="Download all tiles locally first (faster for repeated runs)")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip FAISS database creation, only convert to Parquet")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.tiles_dir and not args.gcs_prefix:
        raise ValueError("Provide either --tiles-dir or --gcs-prefix")
    
    # Handle download-first option
    if args.download_first and args.gcs_prefix:
        if gcsfs is None:
            raise RuntimeError("gcsfs is required for --download-first. Install with: pip install gcsfs")
        
        print("📥 Downloading all tiles from GCS first...")
        download_dir = os.path.join(os.getcwd(), "downloaded_tiles")
        os.makedirs(download_dir, exist_ok=True)
        
        fs = gcsfs.GCSFileSystem()
        prefix = args.gcs_prefix.rstrip('/')
        geojson_files = fs.glob(f"{prefix}/**/*.geojson") or fs.glob(f"{prefix}/*.geojson")
        
        print(f"📦 Found {len(geojson_files)} files to download")
        
        for i, gcs_file in enumerate(geojson_files):
            local_file = os.path.join(download_dir, os.path.basename(gcs_file))
            if not os.path.exists(local_file):
                print(f"  📥 Downloading {i+1}/{len(geojson_files)}: {os.path.basename(gcs_file)}")
                fs.download(gcs_file, local_file)
            else:
                print(f"  ✅ Already exists: {os.path.basename(gcs_file)}")
        
        # Switch to local directory
        args.tiles_dir = download_dir
        args.gcs_prefix = None
        print(f"✅ Downloaded to: {download_dir}")
    
    # Set default paths
    if args.output_parquet is None:
        base_dir = args.tiles_dir if args.tiles_dir else os.getcwd()
        args.output_parquet = os.path.join(base_dir, "embeddings.parquet")
    
    if args.db_dir is None and not args.skip_faiss:
        base_dir = args.tiles_dir if args.tiles_dir else os.getcwd()
        args.db_dir = os.path.join(base_dir, "faiss_db")
    
    # Convert GeoJSON tiles to Parquet
    convert_geojson_tiles_to_parquet(
        tiles_dir=args.tiles_dir,
        output_parquet=args.output_parquet,
        embedding_bands=args.embedding_bands,
        gcs_prefix=args.gcs_prefix,
        max_workers=args.max_workers
    )
    
    # Ingest to FAISS database
    if not args.skip_faiss:
        ingest_to_faiss_db(
            parquet_file=args.output_parquet,
            db_dir=args.db_dir,
            embedding_bands=args.embedding_bands
        )
    
    print("\n🎉 Processing complete!")
    if not args.skip_faiss:
        print(f"📁 FAISS database location: {args.db_dir}")
    print(f"📁 Parquet file: {args.output_parquet}")

if __name__ == "__main__":
    main()
