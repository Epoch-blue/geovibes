#!/usr/bin/env python3
"""
Unified pipeline: GEE → xee/xarray → coordinate points → stream ingest → FAISS index

This script orchestrates the complete workflow for extracting satellite embeddings
from Google Earth Engine, converting them to coordinate-based points, ingesting
into DuckDB, and building a FAISS index for similarity search.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geovibes.database.xee_embeddings import (
    extract_embeddings_to_dataframe,
    extract_embeddings_streaming_generator,
    stream_pixels_to_parquet,
    read_geotiff_streaming
)
from geovibes.database.stream_ingest import (
    setup_stream_logging,
    create_embeddings_table,
    stream_ingest_dataframe,
    stream_ingest_generator,
    ingest_parquet_files_streaming,
    create_rtree_index,
    verify_ingestion
)
from geovibes.database.faiss_db import (
    infer_embedding_dim_from_file,
    create_faiss_index
)
from geovibes.database.xee_embeddings import initialize_earth_engine, load_roi_geometry, get_alphaearth_embeddings, export_embeddings_to_geotiff


def run_xee_pipeline(
    roi_geojson: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    output_dir: str = ".",
    db_name: str = "embeddings",
    scale: int = 100,
    resample_scale: Optional[int] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    batch_size: int = 10000,
    service_account_key: Optional[str] = None,
    skip_faiss: bool = False,
    nlist: int = 4096,
    m: int = 64,
    nbits: int = 8,
    geotiff_mode: bool = False,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    geotiff_path: Optional[str] = None,
    faiss_only: bool = False
) -> dict:
    """
    Run the complete xee-based embedding pipeline.
    
    Supports two modes:
    1. Streaming (default): xarray streaming directly
    2. GeoTIFF (--geotiff-mode): Batch export to GeoTIFF, then ingest locally
    
    Args:
        geotiff_mode: Use batch export to GeoTIFF workflow
        bucket: GCS bucket for GeoTIFF export
        prefix: GCS path prefix for export
        geotiff_path: Path to GeoTIFF file to ingest (gs:// or local)
        ... (other args as before)
        
    Returns:
        Dictionary with pipeline results
    """
    if faiss_only:
        print("\n" + "="*70)
        print("🚀 FAISS Index Creation Only")
        print("="*70)
        
        output_path = Path(output_dir)
        db_path = output_path / f"{db_name}_metadata.db"
        index_path = output_path / f"{db_name}_faiss.index"
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        print(f"📊 Creating FAISS index from existing database: {db_path}")
        
        from geovibes.database.stream_ingest import verify_ingestion
        from geovibes.database.faiss_db import create_faiss_index
        
        stats = verify_ingestion(str(db_path))
        print(f"📈 Database stats: {stats['total_rows']:,} rows, {stats['embedding_dimension']} dimensions")
        
        print("\n" + "-"*70)
        print("Creating FAISS index...")
        print("-"*70)
        
        # Use conservative batch size for memory safety
        memory_safe_batch_size = 50_000  # Very conservative batch size for memory safety
        print(f"Using memory-safe batch size: {memory_safe_batch_size:,} vectors per batch")
        create_faiss_index(str(db_path), str(index_path), stats['embedding_dimension'], "FLOAT", nlist, m, nbits, memory_safe_batch_size)
        
        print(f"\n✅ FAISS index created: {index_path}")
        return {'status': 'faiss_completed', 'db_path': str(db_path), 'index_path': str(index_path)}
    
    if geotiff_mode:
        if geotiff_path:
            print("\n" + "="*70)
            print("🚀 GeoTIFF Ingest Phase")
            print("="*70)
            
            from geovibes.database.xee_embeddings import read_geotiff_streaming
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            setup_stream_logging(os.path.join(output_dir, "xee_pipeline.log"))
            
            db_path = output_path / f"{db_name}_metadata.db"
            index_path = output_path / f"{db_name}_faiss.index"
            parquet_path = output_path / f"{db_name}_embeddings.parquet"
            
            print(f"📁 Output directory: {output_path}")
            print(f"📊 Reading from: {geotiff_path}")
            
            print("\n" + "-"*70)
            print("PHASE 1: Stream read GeoTIFFs and ingest to DuckDB")
            print("-"*70)
            
            create_embeddings_table(str(db_path), drop_existing=True)
            
            total_ingested = 0
            global_id_counter = 0
            for batch_df in read_geotiff_streaming(geotiff_path, tile_id=db_name, batch_size=batch_size):
                ingested = stream_ingest_dataframe(str(db_path), batch_df, batch_size=batch_size, embedding_col='embedding', start_id=global_id_counter)
                total_ingested += ingested
                global_id_counter += len(batch_df)
            
            create_rtree_index(str(db_path))
            stats = verify_ingestion(str(db_path))
            
            print(f"\n✅ Ingestion complete: {total_ingested} points")
            
            if skip_faiss:
                return {'status': 'completed_no_faiss', 'total_rows': total_ingested, 'db_path': str(db_path)}
            
            print("\n" + "-"*70)
            print("PHASE 2: Create FAISS index")
            print("-"*70)
            
            create_faiss_index(str(db_path), str(index_path), stats['embedding_dimension'], "FLOAT", nlist, m, nbits, 1_000_000)
            
            return {'status': 'completed', 'total_rows': total_ingested, 'db_path': str(db_path), 'index_path': str(index_path)}
        
        else:
            print("\n" + "="*70)
            print("🚀 GeoTIFF Export Phase (Batch)")
            print("="*70)
            
            if not bucket or not prefix:
                raise ValueError("Must provide --bucket and --prefix for GeoTIFF export mode")
            
            initialize_earth_engine(service_account_key)
            roi_ee = load_roi_geometry(roi_geojson, bbox)
            image = get_alphaearth_embeddings(roi_ee, start_date, end_date, scale)
            
            export_embeddings_to_geotiff(image, roi_ee, scale, bucket, prefix, start_date, end_date)
            
            return {'status': 'export_queued', 'bucket': bucket, 'prefix': prefix, 'message': 'Export task queued - wait for completion then run ingest phase with --geotiff-path'}
    
    print("\n" + "="*70)
    print("🚀 Starting XEE Embedding Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    setup_stream_logging(os.path.join(output_dir, "xee_pipeline.log"))
    
    db_path = output_path / f"{db_name}_metadata.db"
    index_path = output_path / f"{db_name}_faiss.index"
    parquet_path = output_path / f"{db_name}_embeddings.parquet"
    
    print(f"\n📁 Output directory: {output_path}")
    print(f"📊 Database path: {db_path}")
    print(f"🔍 Index path: {index_path}")
    print(f"📦 Parquet path: {parquet_path}")
    
    print("\n" + "-"*70)
    print("PHASE 1: Extract embeddings from GEE using xee")
    print("-"*70)
    
    phase1_start = time.time()
    
    print("🔄 Using streaming-to-Parquet path for memory efficiency")
    
    df_gen = extract_embeddings_streaming_generator(
        roi_geojson=roi_geojson,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        scale=scale,
        resample_scale=resample_scale,
        service_account_key=service_account_key,
        batch_size=batch_size
    )
    
    total_rows = 0
    all_batches = []
    for batch_df in df_gen:
        all_batches.append(batch_df)
        total_rows += len(batch_df)
        print(f"✅ Received batch: +{len(batch_df)} rows (total: {total_rows})")
    
    if total_rows == 0:
        print("⚠️  No data extracted - Parquet file not created")
        return {'status': 'no_data', 'error': 'No embeddings found in region'}
    
    print(f"📝 Concatenating {len(all_batches)} batches...")
    df = pd.concat(all_batches, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Wrote {len(df)} rows to Parquet")
    
    print(f"✅ Loaded {len(df)} coordinate points from Parquet")
    
    print(f"✅ Extracted {len(df)} coordinate points")
    print(f"   Embedding dimension: {len(df.iloc[0]['embedding'])}")
    print(f"   Bounds: lon [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
    print(f"           lat [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    
    phase1_time = time.time() - phase1_start
    print(f"⏱️  Phase 1 completed in {phase1_time:.2f}s")
    
    print("\n" + "-"*70)
    print("PHASE 2: Create DuckDB table and stream ingest")
    print("-"*70)
    
    phase2_start = time.time()
    
    create_embeddings_table(str(db_path), drop_existing=True)
    
    total_ingested = stream_ingest_dataframe(
        str(db_path),
        df,
        batch_size=batch_size,
        embedding_col='embedding'
    )
    
    create_rtree_index(str(db_path))
    
    stats = verify_ingestion(str(db_path))
    
    phase2_time = time.time() - phase2_start
    print(f"⏱️  Phase 2 completed in {phase2_time:.2f}s")
    
    if skip_faiss:
        print("\n" + "="*70)
        print("⏭️  Skipping FAISS index creation (--skip-faiss flag)")
        print("="*70)
        
        total_time = time.time() - start_time
        
        results = {
            'status': 'completed_no_faiss',
            'total_rows': total_ingested,
            'embedding_dimension': stats['embedding_dimension'],
            'output_dir': str(output_path),
            'db_path': str(db_path),
            'parquet_path': str(parquet_path),
            'total_time_seconds': total_time,
            'bounds': stats['bounds']
        }
        
        print("\n📋 Pipeline Results:")
        for key, value in results.items():
            print(f"   {key}: {value}")
        
        return results
    
    print("\n" + "-"*70)
    print("PHASE 3: Create FAISS index")
    print("-"*70)
    
    phase3_start = time.time()
    
    embedding_dim = stats['embedding_dimension']
    print(f"🎯 Building FAISS index with {embedding_dim}-dimensional embeddings")
    
    create_faiss_index(
        db_path=str(db_path),
        index_path=str(index_path),
        embedding_dim=embedding_dim,
        dtype="FLOAT",
        nlist=nlist,
        m=m,
        nbits=nbits,
        batch_size=1_000_000
    )
    
    phase3_time = time.time() - phase3_start
    print(f"⏱️  Phase 3 completed in {phase3_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Pipeline Completed Successfully!")
    print("="*70)
    
    results = {
        'status': 'completed',
        'total_rows': total_ingested,
        'embedding_dimension': embedding_dim,
        'output_dir': str(output_path),
        'db_path': str(db_path),
        'index_path': str(index_path),
        'parquet_path': str(parquet_path),
        'total_time_seconds': total_time,
        'phase_times': {
            'extraction': phase1_time,
            'ingestion': phase2_time,
            'faiss_index': phase3_time
        },
        'bounds': stats['bounds']
    }
    
    print("\n📋 Pipeline Results:")
    for key, value in results.items():
        if key != 'phase_times':
            print(f"   {key}: {value}")
    
    print("\n⏱️  Timing Summary:")
    for phase, duration in results['phase_times'].items():
        print(f"   {phase}: {duration:.2f}s")
    print(f"   total: {total_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="XEE-based satellite embedding extraction pipeline"
    )
    
    region_group = parser.add_mutually_exclusive_group(required=False)
    region_group.add_argument("--roi-file", help="Path to GeoJSON defining ROI")
    region_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help="Bounding box (decimal degrees)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for database files"
    )
    
    parser.add_argument(
        "--db-name",
        default="embeddings",
        help="Name for the database (default: embeddings)"
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="GEE resolution in meters (default: 100)"
    )
    
    parser.add_argument(
        "--resample-scale",
        type=int,
        help="Optional resampling scale (if different from --scale)"
    )
    
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for ingestion (default: 10000)"
    )
    
    parser.add_argument(
        "--service-account-key",
        help="Path to GCP service account key JSON"
    )
    
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS index creation"
    )
    
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="FAISS IVF nlist parameter (default: 4096)"
    )
    
    parser.add_argument(
        "--m",
        type=int,
        default=64,
        help="FAISS PQ m parameter (default: 64)"
    )
    
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="FAISS PQ nbits parameter (default: 8)"
    )
    
    parser.add_argument(
        "--geotiff-mode",
        action="store_true",
        help="Use batch export to GeoTIFF workflow instead of xee streaming"
    )
    
    parser.add_argument(
        "--bucket",
        type=str,
        help="GCS bucket for GeoTIFF export (required with --geotiff-mode)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        help="GCS path prefix for GeoTIFF export (e.g., embeddings/riau)"
    )
    
    parser.add_argument(
        "--geotiff-path",
        type=str,
        help="Local or gs:// path to GeoTIFF file to ingest (for ingestion phase)"
    )
    
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Only create FAISS index from existing database (skip ingestion)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.faiss_only:
        # FAISS-only mode: just create index from existing database
        if not args.db_name:
            parser.error("Must provide --db-name for --faiss-only mode")
    elif not args.geotiff_path and not args.roi_file and not args.bbox:
        parser.error("Must provide either --roi-file/--bbox (for export) or --geotiff-path (for ingest)")
    
    results = run_xee_pipeline(
        roi_geojson=args.roi_file,
        bbox=tuple(args.bbox) if args.bbox else None,
        output_dir=args.output_dir,
        db_name=args.db_name,
        scale=args.scale,
        resample_scale=args.resample_scale,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        service_account_key=args.service_account_key,
        skip_faiss=args.skip_faiss,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        geotiff_mode=args.geotiff_mode,
        bucket=args.bucket,
        prefix=args.prefix,
        geotiff_path=args.geotiff_path,
        faiss_only=args.faiss_only
    )


if __name__ == "__main__":
    main()
