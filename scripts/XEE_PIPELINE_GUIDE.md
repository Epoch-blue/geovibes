# XEE Embedding Pipeline Guide

## Overview

The XEE (xarray Earth Engine) embedding pipeline provides an efficient end-to-end workflow for extracting satellite embeddings from Google Earth Engine and building searchable FAISS indices.

**Pipeline Architecture:**

```
Google Earth Engine
        ↓
    [Resample embeddings to target scale]
        ↓
    xee/xarray extraction
        ↓
    Convert pixels → coordinate points
        ↓
    Stream ingest to DuckDB
        ↓
    Create FAISS index
        ↓
    Ready for similarity search
```

## Key Advantages Over Previous Approach

1. **Direct xarray/xee extraction**: Eliminates intermediate GeoJSON/Parquet conversion steps
2. **Memory efficient**: Streams data in batches rather than loading everything at once
3. **Pixel-to-coordinate conversion**: Each pixel becomes a searchable point with precise coordinates
4. **Resampling in GEE**: Significantly reduces data volume before extraction
5. **Single unified script**: No need to chain multiple scripts together

## Installation

Ensure xee is installed:

```bash
uv pip install xee
```

## Usage

### Basic Usage (Bounding Box)

```bash
python scripts/xee_pipeline.py \
  --bbox -120.0 40.0 -119.0 41.0 \
  --output-dir ./embeddings_db \
  --db-name california_test \
  --scale 10
```

### With GeoJSON ROI

```bash
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ./embeddings_db \
  --db-name alabama_embeddings \
  --scale 10 \
  --resample-scale 20
```

### With Custom FAISS Parameters

```bash
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ./embeddings_db \
  --db-name alabama_large \
  --scale 10 \
  --nlist 8192 \
  --m 128 \
  --nbits 8
```

### Skip FAISS Index Creation

```bash
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ./embeddings_db \
  --db-name alabama_data_only \
  --scale 10 \
  --skip-faiss
```

### GeoTIFF Export Mode (Batch Export)

For large regions or when you need to export to GeoTIFF first, then ingest locally:

```bash
# Step 1: Export to GeoTIFF in GCS
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ./embeddings_db \
  --db-name alabama \
  --scale 10 \
  --geotiff-mode \
  --bucket your-gcs-bucket \
  --prefix embeddings/alabama
```

This will queue an Earth Engine export task to Google Cloud Storage. After the export completes:

```bash
# Step 2: Ingest from GeoTIFF
python scripts/xee_pipeline.py \
  --geotiff-path gs://your-gcs-bucket/embeddings/alabama/alphaearth_alabama.tif \
  --output-dir ./embeddings_db \
  --db-name alabama \
  --scale 10
```

### FAISS Index Only (From Existing Database)

If you have an existing database and only need to create the FAISS index:

```bash
python scripts/xee_pipeline.py \
  --faiss-only \
  --db-name alabama \
  --output-dir ./embeddings_db \
  --nlist 4096 \
  --m 64 \
  --nbits 8
```

## Command-Line Arguments

### Required (one of):
- `--roi-file PATH`: Path to GeoJSON file defining region of interest
- `--bbox WEST SOUTH EAST NORTH`: Bounding box in decimal degrees
- `--geotiff-path PATH`: Local or `gs://` path to GeoTIFF file to ingest (for ingestion phase)
- `--faiss-only`: Create FAISS index from existing database (requires `--db-name`)

### Optional:
- `--output-dir PATH`: Directory for output database files (default: current directory)
- `--db-name NAME`: Name for the database (default: "embeddings")
- `--scale METERS`: GEE resolution in meters (default: 100)
- `--resample-scale METERS`: Optional resampling scale (if different from --scale)
- `--start-date YYYY-MM-DD`: Start date (default: 2024-01-01)
- `--end-date YYYY-MM-DD`: End date (default: 2024-12-31)
- `--batch-size N`: Batch size for streaming operations (default: 10000)
- `--service-account-key PATH`: Path to GCP service account key JSON
- `--skip-faiss`: Skip FAISS index creation
- `--nlist N`: FAISS IVF nlist parameter (default: 4096)
- `--m N`: FAISS PQ m parameter (default: 64)
- `--nbits N`: FAISS PQ nbits parameter (default: 8)
- `--geotiff-mode`: Use batch export to GeoTIFF workflow instead of xee streaming (requires `--bucket` and `--prefix`)
- `--bucket BUCKET`: GCS bucket for GeoTIFF export (required with `--geotiff-mode`)
- `--prefix PREFIX`: GCS path prefix for GeoTIFF export (e.g., `embeddings/riau`, required with `--geotiff-mode`)
- `--geotiff-path PATH`: Local or `gs://` path to GeoTIFF file to ingest (alternative to `--roi-file`/`--bbox` for ingestion)

## Data Pipeline

The pipeline supports two modes:

### Mode 1: Direct Streaming (Default)

All extractions use the streaming-to-Parquet pathway for optimal memory efficiency:

```
Google Earth Engine
        ↓
  [Load AlphaEarth embeddings]
        ↓
  [Resample to target scale in GEE]
        ↓
  xee extraction → xarray DataArray
        ↓
  Convert pixels to coordinate points (batches)
        ↓
  Stream to Parquet (intermediate checkpoint)
        ↓
  Stream ingest from Parquet → DuckDB
        ↓
  Create R-Tree spatial index
        ↓
  Create FAISS IVF-PQ index
        ↓
  Ready for similarity search
```

### Mode 2: GeoTIFF Export (For Large Regions)

For very large regions or when you need to export first and ingest later:

```
Google Earth Engine
        ↓
  [Load AlphaEarth embeddings]
        ↓
  [Resample to target scale in GEE]
        ↓
  Batch export to GeoTIFF → GCS
        ↓
  [Wait for export task to complete]
        ↓
  Stream read GeoTIFF (local or gs://)
        ↓
  Convert pixels to coordinate points (batches)
        ↓
  Stream ingest to DuckDB
        ↓
  Create R-Tree spatial index
        ↓
  Create FAISS IVF-PQ index
        ↓
  Ready for similarity search
```

### Tiling for Large Regions

For ROI extents larger than ~5°x5°, the pipeline automatically tiles the region into smaller 5°x5° chunks. Each tile is:
- Extracted independently 
- Converted to coordinate points
- Streamed to Parquet
- Labeled with tile index for tracking

This approach:
- ✅ Avoids memory overflows from large extractions
- ✅ Handles API size limits gracefully
- ✅ Enables parallel processing (future enhancement)
- ✅ Maintains consistent memory usage

**Benefits of this approach:**
- ✅ Memory-efficient batching throughout
- ✅ Parquet checkpoint for resumability
- ✅ Consistent behavior across all region sizes
- ✅ Intermediate data can be inspected/queried independently
- ✅ Cloud/serverless deployment friendly

## Pipeline Phases

### Phase 1: Extract Embeddings from GEE

- Loads AlphaEarth embeddings from Google Earth Engine
- Optionally resamples to specified scale
- Extracts as xarray DataArray using xee
- Converts pixels to coordinate-based points
- Saves as Parquet for reference

**Output files:**
- `{db_name}_embeddings.parquet` - Extracted embeddings with coordinates

### Phase 2: Stream Ingest to DuckDB

- Creates geo_embeddings table in DuckDB
- Stream ingests points in batches
- Creates R-Tree spatial index on geometry
- Verifies ingestion and reports statistics

**Output files:**
- `{db_name}_metadata.db` - DuckDB database with spatial data

### Phase 3: Create FAISS Index

- Builds optimized FAISS IVF-PQ index
- Stores index separately for efficient similarity search
- Uses specified parameters (nlist, m, nbits)

**Output files:**
- `{db_name}_faiss.index` - FAISS index for fast nearest-neighbor search

## Python API

### Extract embeddings programmatically

```python
from geovibes.database.xee_embeddings import extract_embeddings_to_dataframe

df = extract_embeddings_to_dataframe(
    roi_geojson="geometries/alabama.geojson",
    scale=10,
    resample_scale=20,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"Extracted {len(df)} points")
print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
```

### Stream ingest and build FAISS

```python
from geovibes.database.stream_ingest import (
    create_embeddings_table,
    stream_ingest_dataframe,
    verify_ingestion
)
from geovibes.database.faiss_db import create_faiss_index

db_path = "local_databases/alabama_metadata.db"
index_path = "local_databases/alabama_faiss.index"

create_embeddings_table(db_path)
total_rows = stream_ingest_dataframe(db_path, df)
stats = verify_ingestion(db_path)

create_faiss_index(
    db_path=db_path,
    index_path=index_path,
    embedding_dim=stats['embedding_dimension'],
    dtype="FLOAT",
    nlist=4096,
    m=64,
    nbits=8
)
```

### Complete pipeline in Python

```python
from scripts.xee_pipeline import run_xee_pipeline

results = run_xee_pipeline(
    roi_geojson="geometries/alabama.geojson",
    output_dir="./embeddings_db",
    db_name="alabama",
    scale=10
)

print(f"Total rows: {results['total_rows']}")
print(f"Total time: {results['total_time_seconds']:.2f}s")
print(f"Phase times: {results['phase_times']}")
```

## Output Files

For each pipeline run with `--db-name myregion`, you'll get:

1. **myregion_metadata.db** - DuckDB database containing:
   - `geo_embeddings` table with (id, lon, lat, embedding, geometry, tile_id)
   - R-Tree spatial index for efficient spatial queries
   
2. **myregion_faiss.index** - FAISS index for fast similarity search

3. **myregion_embeddings.parquet** - Extracted embeddings (for reference/debugging)

4. **xee_pipeline.log** - Detailed execution log

## Earth Engine Authentication

The pipeline handles Earth Engine authentication automatically:

1. **If already authenticated:**
   ```bash
   earthengine authenticate
   ```

2. **Using service account key:**
   ```bash
   python scripts/xee_pipeline.py \
     --service-account-key /path/to/key.json \
     ...
   ```

## Workflow Modes

### Standard Streaming Mode (Default)

The default mode uses xee to stream data directly from Google Earth Engine. This is the most memory-efficient approach and works well for most regions:

```bash
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ./embeddings_db \
  --db-name alabama
```

### GeoTIFF Export Mode

For very large regions (continental scale) or when you need to export first and ingest later (e.g., on a different machine), use `--geotiff-mode`:

```bash
# Export phase
python scripts/xee_pipeline.py \
  --roi-file geometries/large_region.geojson \
  --geotiff-mode \
  --bucket your-gcs-bucket \
  --prefix embeddings/large_region \
  --scale 50

# Wait for Earth Engine export to complete, then ingest
python scripts/xee_pipeline.py \
  --geotiff-path gs://your-gcs-bucket/embeddings/large_region/alphaearth_large_region.tif \
  --output-dir ./embeddings_db \
  --db-name large_region \
  --scale 50
```

**Benefits of GeoTIFF mode:**
- ✅ Handles very large regions that may exceed GEE API limits in streaming mode
- ✅ Allows export and ingestion on different machines
- ✅ Provides a reusable GeoTIFF checkpoint
- ✅ Can be resumed if ingestion fails

### FAISS Index Only Mode

If you have an existing database (created previously or from another source) and only need to create the FAISS index:

```bash
python scripts/xee_pipeline.py \
  --faiss-only \
  --db-name existing_database \
  --output-dir ./embeddings_db \
  --nlist 8192 \
  --m 128 \
  --nbits 8
```

This is useful when:
- You want to rebuild the index with different parameters
- The database was created without the index (`--skip-faiss`)
- You're optimizing FAISS parameters for your use case