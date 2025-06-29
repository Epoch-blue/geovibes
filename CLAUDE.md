# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Code style
Do not add comments unless specifically asked to.
Do not wrap single functionality in a function, i.e 'load_csv' should just expose pd.read_csv

## Project Overview

GeoVibes is a geospatial similarity search tool that leverages satellite foundation model embeddings for "vibe checking" geographic models. It provides an interactive map interface for labeling points and performing similarity searches across large-scale embedding datasets.

## Architecture

The system follows a layered architecture:

- **UI Layer**: Interactive Jupyter notebook interface (`vibe_checker.ipynb`) with ipyleaflet maps
- **Core Processing**: GeoVibes class (`src/ui.py`) handles labeling, search logic, and query vector computation
- **Database Layer**: DuckDB with HNSW vector index and RTree spatial index for efficient similarity search
- **Data Pipeline**: Scripts for downloading Earth Genome embeddings and building searchable databases

Key components:
- `src/ui.py`: Main GeoVibes interface class with interactive map labeling
- `src/earth_genome/duckdb_embedding_index.py`: Database creation with vector and spatial indexing
- `src/earth_genome/region_to_eg_embeddings.py`: Downloads embeddings from Earth Genome S3 bucket
- `config/vibes_config.json`: Configuration file specifying database path, boundary, and date range

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
mamba create -n geovibes python=3.12 -y
mamba activate geovibes
mamba install -c conda-forge --file ./requirements.txt -y
```

### Database Creation Workflow
```bash
# 1. Download embeddings for a region
python src/earth_genome/region_to_eg_embeddings.py region.geojson \
  --filter-land-only \
  --mgrs-reference-file geometries/MGRS_LAND.geojson \
  --out-dir embeddings/my_region

# 2. Build DuckDB database with HNSW index
python src/earth_genome/duckdb_embedding_index.py embeddings/my_region output.db
```

### Running the Interface
The primary interface is `vibe_checker.ipynb` which requires:
- A `.env` file with `MAPTILER_API_KEY="your-api-key"`
- A config file (see `config/vibes_config.json` for format)

## Key Design Patterns

### Memory Management
- Embeddings are cached in `cached_embeddings` dict to avoid repeated database queries
- Chunked embedding fetching (default 1000 embeddings per chunk) for large polygon selections
- Light queries without embeddings for spatial operations, followed by on-demand embedding fetch

### Query Vector Computation
Query vectors use the formula: `2 * positive_average - negative_average`
- Positive labels get averaged and weighted by 2
- Negative labels get averaged and subtracted
- Supports iterative refinement through additional labeling

### Database Schema
The `geo_embeddings` table contains:
- `id`: Point identifier (string or integer)
- `geometry`: Spatial point geometry
- `embedding`: High-dimensional vector (typically 384 dimensions)

Indexes:
- HNSW index on `embedding` column for vector similarity search
- RTree index on `geometry` column for spatial queries

## File Structure Notes

- `src/ui_config/`: UI constants, basemap configurations, and database settings
- `geometries/`: Boundary files and MGRS reference data
- `databases/`: DuckDB files with indexed embeddings
- `embeddings/`: Raw embedding parquet files from Earth Genome
- `notebooks/`: Jupyter notebooks for analysis and experimentation

## Dependencies

Key packages:
- `duckdb>1.0.0`: Database with vector search extensions
- `geopandas`: Geospatial data processing
- `ipyleaflet`: Interactive maps in Jupyter
- `shapely`: Geometric operations
- `earthengine-api`: Optional, for NDVI/NDWI basemaps
- `pyarrow`: Parquet file handling

## Testing and Quality

This repository does not currently have automated tests or linting configuration. Code quality is maintained through manual review.