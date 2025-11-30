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

The script lets you select regions to download, stores geometries in `geometries/`, and extracts the model artifacts into `local_databases/`. You can rerun it at any time; previously downloaded files are skipped. We recommend starting with a small database: for example a google database, or a quantized one.

## Configuration

### YAML Configuration (Recommended)

Create a `config.yaml` file to configure GeoVibes. Paths to DuckDB databases and boundaries are optional—GeoVibes now discovers downloaded models automatically via `manifest.csv`:

```yaml
# Basemap date range
start_date: "2024-01-01"
end_date: "2025-01-01"

```

### Environment Variables
The main basemap relies on MapTiler. You need to create a [MapTiler account](https://cloud.maptiler.com/maps/) and get an API key in order for this to work.
Create a `.env` file in the repository root for sensitive configuration:

```env
# Required for MapTiler satellite basemaps
MAPTILER_API_KEY=your_maptiler_api_key_here
```

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

#### Interactive Search Examples

**Label a point and search**  
Start your search by picking a point for which you would like to find similar ones in your area, then click Search. This will open a tile panel showing your search results but also show them on the map.

<img src="images/label_and_search.gif" alt="Label a point and search for similar points" width="700" />


**Polygon Labeling**  
Search is iterative: positives get added to your query vector and negatives get subtracted. Use polygon labeling mode for bulk positive/negative selection.
<img src="images/polygon_labels.gif" alt="Polygon labeling and search for similar points" width="700" />


**Tile Panel Labeling**
You can use the tile panel to pan the map to a given tile, as well as label them.

<img src="images/label_search_tile_panel.gif" alt="Tile panning and labeling" width="700" />

**Change basemap**
You can cycle through different basemaps as you are searching. This will cycle through in on both the leaflet map and the individual tiles in the tile panel. NB: we currently do not support pulling the S2 basemaps into the tiles, these are only available as tiles on the leaflet map served via Google Earth Engine.

<img src="images/label_search_basemap_change.gif" alt="Cycling through basemaps" width="700" />


**Load Previous Datasets**  
Save your search results as GeoJSON and reload them to continue searching.

<img src="images/load_dataset.gif" alt="Load a previous dataset" width="700" />


**Use Google Street View**
You can also use google maps/street view to help you label.

<img src="images/gsv.gif" alt="Google Street View" width="700" />

## Architecture

The GeoVibes system is designed for efficient large-scale geospatial similarity search. The core of the system is a hybrid architecture combining a FAISS index for fast vector search with a DuckDB database for storing metadata and geometries.

This dual-component system is built using the script in `geovibes/database/faiss_db.py` and is optimized for performance with the following features:

- **FAISS Index:** It uses a highly optimized FAISS (Facebook AI Similarity Search) index for efficient similarity search on high-dimensional vector embeddings. The script builds an Inverted File (IVF) index with Product Quantization (PQ) for float embeddings or Scalar Quantization for int8 embeddings, enabling fast approximate nearest neighbor search.
- **DuckDB Metadata Store:** A DuckDB database stores metadata associated with each vector, including a unique ID and the geometry. This allows for quick retrieval of spatial and other attributes after a vector search.

This combination of a dedicated vector index and a spatial database allows GeoVibes to perform complex queries that combine both content-based similarity and geographic location.


### Earth Engine Authentication (Optional - for NDVI/NDWI basemaps)

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
- Passing `enable_ee=True` when you construct `GeoVibes`
- Exporting `GEOVIBES_ENABLE_EE=1` in your environment

### Step 3: Build Searchable Database

This method uses Facebook AI's FAISS library to create a highly optimized index, which is stored separately from the metadata in a DuckDB database.
you can point this script at either a local set of embeddings or embeddings stored in the cloud. In this example we use the [Earth Genome Softcon Embeddings on Source Cooperative](https://source.coop/earthgenome/earthindexembeddings/2024).

```bash
# Build a full Alabama index using a ROI and remote embeddings
python geovibes/database/faiss_db.py \
  --roi-file geometries/alabama.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://us-west-2.opendata.source.coop/earthgenome/earthindexembeddings/2024/ \
  --name earthgenome_softcon_2024_2025 \
  --tile-pixels 32 \
  --tile-overlap 16 \
  --tile-resolution 10 \
  --output_dir local_databases

# Build a smaller dry-run index for testing
python geovibes/database/faiss_db.py \
  --roi-file geometries/alabama.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://us-west-2.opendata.source.coop/earthgenome/earthindexembeddings/2024/ \
  --name earthgenome_softcon_2024_2025 \
  --tile-pixels 32 --tile-overlap 16 --tile-resolution 10 \
  --output_dir local_databases \
  --dry-run --dry-run-size 5
```

This script processes parquet files from an input directory, builds a FAISS index, and creates a separate DuckDB file containing the metadata for the embeddings.

## Classification Pipeline

GeoVibes includes a classification pipeline for training binary classifiers on labeled embeddings. This enables workflows like: label examples → train classifier → review detections → refine with corrections.

### Basic Usage

```bash
# Train classifier on labeled data
uv run python -m geovibes.classification.pipeline \
    --positives labeled_dataset.geojson \
    --output results/ \
    --db path/to/embeddings.duckdb \
    --threshold 0.5
```

### Iterative Retraining with Corrections

After reviewing detection results in the GeoVibes UI, export corrections and retrain:

```bash
# Combine original labels with corrections from detection review
uv run python -m geovibes.classification.pipeline \
    --positives original_labels.geojson corrections.geojson \
    --output results_v2/ \
    --db path/to/embeddings.duckdb \
    --correction-weight 2.5
```

The `--correction-weight` parameter emphasizes correction samples (class `relabel_pos`/`relabel_neg`) during training, helping the model learn from its mistakes.

### Detection Review Mode

When you load a GeoJSON file with a `probability` field (classifier output), GeoVibes automatically enters detection review mode:

- **Threshold slider**: Filter detections by probability, with min/max set from your dataset
- **Colormap visualization**: Probability-scaled colors (plasma colormap)
- **Interactive labeling**: Click or polygon-select to mark false positives/negatives
- **Export corrections**: Save as augmented dataset for retraining

### Spatial Cross-Validation

For robust model evaluation, use 5-fold spatial cross-validation instead of a single train/test split:

```bash
uv run python -m geovibes.classification.pipeline \
    --positives labeled_data.geojson corrections.geojson \
    --negatives sampled_negatives.geojson \
    --db path/to/embeddings.duckdb \
    --output results/ \
    --cv \
    --cv-buffer 500
```

The spatial CV algorithm:
1. Buffers positive points (default 500m) and unions into contiguous clusters
2. Explodes union into separate polygon regions
3. Distributes clusters across 5 folds with greedy sample balancing
4. Assigns negatives to the fold of their nearest positive cluster
5. Reports mean ± std for all metrics

Example output:
```
=================================================================
SPATIAL 5-FOLD CROSS-VALIDATION RESULTS
=================================================================
Metric       Mean ± Std
-----------------------------------------------------------------
Accuracy:    0.9734 ± 0.0297
Precision:   0.9387 ± 0.0988
Recall:      0.9805 ± 0.0115
F1:          0.9566 ± 0.0590
AUC-ROC:     0.9970 ± 0.0034
-----------------------------------------------------------------
Spatial clusters: 59 (distributed across 5 folds)
=================================================================
```

This tests generalization to **new geographic areas**, not just new samples from areas the model has seen.

### CLI Options

| Argument | Description |
|----------|-------------|
| `--positives` | One or more GeoJSON files with training examples |
| `--negatives` | Optional separate file for negative examples |
| `--output` | Output directory for results |
| `--db` | Path to DuckDB database with embeddings |
| `--threshold` | Probability threshold for detection (default: 0.5) |
| `--correction-weight` | Weight multiplier for correction samples (default: 1.0) |
| `--batch-size` | Batch size for inference (default: 100000) |
| `--test-fraction` | Fraction of data for test set (default: 0.2) |
| `--cv` | Run 5-fold spatial cross-validation (skips inference) |
| `--cv-buffer` | Buffer distance in meters for spatial clustering (default: 500) |

### Output Files

The pipeline generates:
- `classification_detections.geojson` - All detections above threshold
- `classification_union.geojson` - Merged detection polygons
- `model.json` - Trained XGBoost model (can be reloaded)

## Performance & Limitations

- **Database scaling**: Tested up to 5M embeddings
- **Memory management**: Two-layer approach for handling large datasets
    - **DuckDB memory limits**: Default 24GB (`MEMORY_LIMIT = '24GB'`) controls database operations
    - **Application chunking**: 10,000 embeddings per chunk prevents Python memory overflow during data transfer
    - Both are necessary: DuckDB limits control internal operations, chunking controls Python data loading
    - Modify `DatabaseConstants.MEMORY_LIMIT` and `EMBEDDING_CHUNK_SIZE` for custom allocation
- **Index performance**: FAISS index creation time varies depending on the number of vectors and parameters. For example, training on a sample of a few million vectors can take several minutes.
- **Future work**: Investigating external vector databases (e.g., Qdrant) and custom embedding pipelines.

## Contributing

GeoVibes is experimental research code. Contributions welcome for:

- Alternative vector index backends (lanceDB etc...)
- Front-end/UI/UX (help please)
- Custom embedding model support
- Performance optimizations
- Documentation improvements

Contact: chris@demeterlabs.io
