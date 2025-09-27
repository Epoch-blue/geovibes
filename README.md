# Evaluate Your Geospatial Models Vibes

Yeah benchmarks are cool and stuff, but how are your model's vibes? With this tooling you'll hopefully be able to see via the magic of search/retrieval from your laptop!

This repo was originally inspired by the [Earth Genome notebook tooling](https://github.com/earth-genome/ei-notebook). GeoVibes supports comparison of multiple embeddings models through nearest neighbors search. It supports Google's [AlphaEarth](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) embeddings as well as other benchmarks such as ViT and Clay.

**Highly experimental. This repo is not production-grade code.**

## Installation

GeoVibes can be installed using `uv`.

```bash
# 1) Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# 3) Install GeoVibes in editable mode
uv pip install -e .

# 4) Register a Jupyter kernel for the notebook app
python -m ipykernel install --user --name geovibes --display-name "Python (geovibes)"
```

## Quick Start

### Prepare Local Data

With your virtual environment active, run the interactive downloader to fetch geometries and DuckDB/FAISS bundles listed in `manifest.csv`:

```bash
uv run download_embeddings.py
```

The script lets you select regions to download, stores geometries in `geometries/`, and extracts the model artifacts into `local_databases/`. You can rerun it at any time; previously downloaded files are skipped.

### Notebook (`vibe_checker.ipynb`)

Launch Jupyter Lab (or Notebook) and open `vibe_checker.ipynb`:

```bash
uv run jupyter lab
```

When the notebook opens, select the `Python (geovibes)` kernel registered during installation. The notebook loads the complete GeoVibes UI and expects the assets downloaded by `download_embeddings.py` in `local_databases/` and `geometries/`.

### Features

- **Multiple basemaps**: MapTiler satellite, Sentinel-2 RGB/NDVI/NDWI composites, Google Hybrid maps
- **Flexible labeling**: Point-click and polygon selection for positive/negative examples
- **Iterative search**: Query vector updates with each labeling iteration using `2×positive_avg - negative_avg`
- **Save/load**: Persist labeled datasets as GeoJSON for continued refinement
- **Memory efficient**: Cached embeddings and chunked database queries for large regions

#### Interactive Search Examples

**Label a point and search**  
Start your search by picking a point for which you would like to find similar ones in your area, then click Search
![Label a point and search for similar points](images/label_positive_point.gif)

**Polygon Labeling**  
Search is iterative: positives get added to your query vector and negatives get subtracted. Use polygon labeling mode for bulk positive/negative selection.
![Polygon labeling and search for similar points](images/polygon_label.gif)

**Load Previous Datasets**  
Save your search results as GeoJSON and reload them to continue searching.
![Load a previous dataset](images/load_saved_changes.gif)

## Configuration

### YAML Configuration (Recommended)

Create a `config.yaml` file to configure GeoVibes. Paths to DuckDB databases and boundaries are optional—GeoVibes now discovers downloaded models automatically via `manifest.csv`:

```yaml
# Basemap date range
start_date: "2024-01-01"
end_date: "2025-01-01"

```

### Environment Variables

Create a `.env` file in the repository root for sensitive configuration:

```env
# Required for MapTiler satellite basemaps
MAPTILER_API_KEY=your_maptiler_api_key_here

# Optional: For Google Cloud Storage database access
GCS_ACCESS_KEY_ID=your_access_key_here
GCS_SECRET_ACCESS_KEY=your_secret_key_here
```

## Architecture

The GeoVibes system is designed for efficient large-scale geospatial similarity search. The core of the system is a hybrid architecture combining a FAISS index for fast vector search with a DuckDB database for storing metadata and geometries.

This dual-component system is built using the script in `geovibes/database/faiss_db.py` and is optimized for performance with the following features:

- **FAISS Index:** It uses a highly optimized FAISS (Facebook AI Similarity Search) index for efficient similarity search on high-dimensional vector embeddings. The script builds an Inverted File (IVF) index with Product Quantization (PQ) for float embeddings or Scalar Quantization for int8 embeddings, enabling fast approximate nearest neighbor search.
- **DuckDB Metadata Store:** A DuckDB database stores metadata associated with each vector, including a unique ID and the geometry. This allows for quick retrieval of spatial and other attributes after a vector search.
- **R-Tree Index:** A spatial index (R-Tree) is built on the geometries within DuckDB. This allows for fast spatial querying, like finding all points within a drawn polygon.

This combination of a dedicated vector index and a spatial database allows GeoVibes to perform complex queries that combine both content-based similarity and geographic location.

### 3. Earth Engine Authentication (Optional - for NDVI/NDWI basemaps)

Earth Engine authentication is **completely optional**. GeoVibes works perfectly without it!

If you want to use NDVI and NDWI basemaps, you'll need to authenticate with Google Earth Engine and explicitly opt in to the Earth Engine tiles:

```bash
# Make sure dependencies are installed
uv pip install -e .

# Authenticate with Earth Engine
earthengine authenticate
```

Follow the authentication flow in your browser. This is only required if you want the NDVI/NDWI basemap options. GeoVibes keeps Earth Engine disabled unless you opt in via:

- Adding `enable_ee: true` to your `config.yaml`
- Exporting `GEOVIBES_ENABLE_EE=1` in your environment

### 4. Google Cloud Storage Database Access (Optional - for GCS databases)

GeoVibes can connect to DuckDB databases stored on Google Cloud Storage. If your database is hosted on GCS (e.g., `gs://your-bucket/database.db`), you'll need to set up authentication.

#### Option 1: HMAC Keys (Recommended)

1. **Create HMAC Keys in GCP Console:**
    - Go to [Cloud Storage Settings](https://console.cloud.google.com/storage/settings)
    - Click "Interoperability" tab
    - Click "Create a key" under "Access keys for your user account"
    - Save the Access Key and Secret

2. **Set Environment Variables:**

    ```bash
    export GCS_ACCESS_KEY_ID="your_access_key_here"
    export GCS_SECRET_ACCESS_KEY="your_secret_key_here"
    ```

3. **Or create a `.env` file:**
    ```env
    GCS_ACCESS_KEY_ID=your_access_key_here
    GCS_SECRET_ACCESS_KEY=your_secret_key_here
    MAPTILER_API_KEY=your_maptiler_api_key_here
    ```

#### Option 2: Default Google Cloud Authentication

If you're running on Google Cloud or have `gcloud` configured:

```bash
gcloud auth application-default login
```

#### Security Notes

- **Never commit credentials to version control**
- Add `.env` to your `.gitignore` file
- Use environment variables in production
- Consider using Google Cloud IAM roles for more secure access

## Generate Embeddings

GeoVibes provides code to export embeddings from AlphaEarth via Google Earth Engine. This is a 3-step workflow:

### Step 1: Create tiling assets in GEE

Generate spatial grid tiles for your region and upload them as GEE assets:

```bash
python src/google/tiling_to_gee_asset.py \
  --input_file geometries/mgrs_tiles.parquet \
  --roi_file aoi.geojson \
  --gcs_bucket your-bucket \
  --gee_asset_path projects/your-project/assets/tiles \
  --tilesize 25 \
  --overlap 0 \
  --resolution 10.0
```

This creates a grid of spatial tiles, uploads them to Google Cloud Storage, and imports them as GEE table assets.

### Step 2: Generate embeddings from satellite imagery

Extract embeddings for each tile using Google's satellite embedding model:

```bash
python src/google/embeddings.py \
  --roi_file aoi.geojson \
  --mgrs_reference_file geometries/mgrs_tiles.parquet \
  --year 2024 \
  --gcs_bucket your-bucket \
  --gcs_prefix embeddings/google_satellite_v1 \
  --gee_asset_path projects/your-project/assets/tiles
```

This processes each tile through Google's satellite embedding model and exports results to GCS.

### Step 3: Build Searchable Database

This method uses Facebook AI's FAISS library to create a highly optimized index, which is stored separately from the metadata in a DuckDB database.
Create a FAISS index and a corresponding metadata database from local parquet files:

```bash
# Create a full index from embeddings in a directory
python geovibes/database/faiss_db.py embeddings/eg_nm --name new-mexico --output_dir faiss_db --dtype INT8

# Create a smaller index for testing (dry run)
python geovibes/database/faiss_db.py embeddings/eg_nm --name new-mexico-dry-run --output_dir faiss_db --dtype INT8 --dry-run
```

This script processes parquet files from an input directory, builds a FAISS index, and creates a separate DuckDB file containing the metadata for the embeddings.

## Prerequisites for Google Embeddings

The Google workflow requires:

- **Google Earth Engine account** with authentication
- **Google Cloud Storage bucket** for intermediate file storage
- **gcloud CLI** installed and authenticated

```bash
# Authenticate with Earth Engine
earthengine authenticate

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project your-project-id
```

## Performance & Limitations

- **Database scaling**: Tested up to 3.5M embeddings; 10M+ may cause performance issues
- **Memory management**: Two-layer approach for handling large datasets
    - **DuckDB memory limits**: Default 12GB (`MEMORY_LIMIT = '12GB'`) controls database operations
    - **Application chunking**: 10,000 embeddings per chunk prevents Python memory overflow during data transfer
    - Both are necessary: DuckDB limits control internal operations, chunking controls Python data loading
    - Modify `DatabaseConstants.MEMORY_LIMIT` and `EMBEDDING_CHUNK_SIZE` for custom allocation
- **Index performance**: FAISS index creation time varies depending on the number of vectors and parameters. For example, training on a sample of a few million vectors can take several minutes.
- **Future work**: Investigating external vector databases (e.g., Qdrant) and custom embedding pipelines.

## Contributing

GeoVibes is experimental research code. Contributions welcome for:

- Alternative vector index backends (FAISS, Qdrant)
- Custom embedding model support
- Performance optimizations
- Documentation improvements

Contact: chris@demeterlabs.io
