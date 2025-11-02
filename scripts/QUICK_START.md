# XEE Pipeline Quick Reference

## Installation
```bash
uv pip install xee
earthengine authenticate
```

## One-Liner Examples

### Smallest (test)
```bash
python scripts/xee_pipeline.py --bbox -120 40 -119.5 40.5 --scale 100
```

### Small region
```bash
python scripts/xee_pipeline.py --roi-file geometries/alabama.geojson --scale 10
```

### Large region, coarse resolution
```bash
python scripts/xee_pipeline.py --roi-file geometries/alabama.geojson --scale 50 --resample-scale 100
```

### Skip FAISS, just create database
```bash
python scripts/xee_pipeline.py --roi-file geometries/alabama.geojson --skip-faiss
```

### Create FAISS index from existing database
```bash
python scripts/xee_pipeline.py --faiss-only --db-name alabama --output-dir ./embeddings_db
```

### GeoTIFF export mode (for large regions)
```bash
# Export phase
python scripts/xee_pipeline.py --roi-file geometries/alabama.geojson --geotiff-mode --bucket my-bucket --prefix embeddings/alabama

# After export completes, ingest phase
python scripts/xee_pipeline.py --geotiff-path gs://my-bucket/embeddings/alabama/alphaearth_alabama.tif --db-name alabama
```

### Custom output location and database name
```bash
python scripts/xee_pipeline.py \
  --roi-file geometries/alabama.geojson \
  --output-dir ~/data/embeddings \
  --db-name alabama_v2
```

## All Options
```bash
python scripts/xee_pipeline.py --help
```

## Expected Output
```
output_dir/
├── {db_name}_metadata.db          ← Use with GeoVibes
├── {db_name}_faiss.index          ← Use with GeoVibes
├── {db_name}_embeddings.parquet   ← Checkpoint
└── xee_pipeline.log               ← Debug info
```

## Copy to GeoVibes
```bash
cp output_dir/*.db local_databases/
cp output_dir/*.index local_databases/
```

## Python Usage
```python
from scripts.xee_pipeline import run_xee_pipeline

results = run_xee_pipeline(
    roi_geojson="geometries/alabama.geojson",
    output_dir="./embeddings_db",
    db_name="alabama"
)
```

## Key Parameters

| Flag | Default | Use |
|------|---------|-----|
| `--scale` | 100m | Resolution of extraction |
| `--resample-scale` | (none) | Optional second resampling step |
| `--batch-size` | 10000 | Rows per streaming batch (↑ for speed, ↓ for memory) |
| `--nlist` | 4096 | FAISS index parameter |
| `--skip-faiss` | false | Skip FAISS index creation |

## Troubleshooting

**"xee" not found?**
```bash
uv pip install xee
```

**Auth error?**
```bash
earthengine authenticate
```

**Out of memory?**
```bash
# Option 1: Larger scale = fewer points
--scale 50

# Option 2: Smaller batches
--batch-size 5000
```

## Performance Estimates

| Scale | Region Size | Points | Time | Memory |
|-------|-------------|--------|------|--------|
| 10m | 10,000 km² | ~1M | 5-10m | ~500MB |
| 20m | 100,000 km² | ~500K | 3-5m | ~500MB |
| 50m | 1M km² | ~100K | 1-2m | ~500MB |

Memory usage is constant (batch-dependent) thanks to streaming architecture

## What Happens

1. **Phase 1** (~30%)
   - Load AlphaEarth from GEE
   - Resample if specified
   - Extract via xee to xarray
   - Convert pixels → coordinate points

2. **Phase 2** (~20%)
   - Create DuckDB table
   - Stream ingest points
   - Create R-Tree spatial index

3. **Phase 3** (~50%)
   - Build FAISS IVF-PQ index
   - Optimize for similarity search

## Next

👉 Copy `.db` and `.index` files to `local_databases/`

👉 Use with GeoVibes UI for interactive search

👉 Query programmatically via DuckDB + FAISS
