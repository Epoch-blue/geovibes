# Data Formats

GeoVibes uses several standardized data formats for storing and exchanging labeled data, detection results, and embeddings.

---

## GeoJSON Format Types

The system recognizes three types of GeoJSON files based on their properties:

| Type | Detection Logic | Purpose |
|------|-----------------|---------|
| **Labeled Dataset** | `label` + `embedding` present | Training data with manual labels |
| **Detection Results** | `probability` present | Classifier predictions for review |
| **Vector Layer** | Neither above | Generic overlay (boundaries, etc.) |

Detection logic is in `datasets.py:detect_geojson_type()`.

---

## 1. Labeled Dataset (Training Data)

Used to save/load labeled points for training classifiers.

### Structure

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "timestamp": "20251212_190027",
    "total_points": 150,
    "positive_points": 100,
    "negative_points": 50,
    "embedding_dimension": 768
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-86.5, 32.3]
      },
      "properties": {
        "id": "12345",
        "tile_id": "abc_123_456",
        "label": 1,
        "class": "geovibes_pos",
        "embedding": [0.1, 0.2, ...],
        "source": "manual"
      }
    }
  ]
}
```

### Properties Reference

| Property | Type | Values | Description |
|----------|------|--------|-------------|
| `id` | string | Database ID | Primary key from `geo_embeddings` table |
| `tile_id` | string | Optional | Original tile identifier from source |
| `label` | int | `1` or `0` | 1 = positive, 0 = negative |
| `class` | string | See below | Semantic label for classification |
| `embedding` | float[] | 768/1024-dim | Feature vector from foundation model |
| `source` | string | `"manual"` or `"detection_review"` | How the label was created |

### Class Values

| Value | Source | Meaning |
|-------|--------|---------|
| `geovibes_pos` | Manual labeling | User marked as positive in search mode |
| `geovibes_neg` | Manual labeling | User marked as negative in search mode |
| `relabel_pos` | Detection review | Confirmed a classifier prediction |
| `relabel_neg` | Detection review | Rejected a false positive |

### Export Functions

- **Basic save**: `DatasetManager.save_dataset()` → `labeled_dataset_{timestamp}.geojson`
- **Augmented save**: `DatasetManager.export_augmented_dataset()` → `augmented_dataset_{timestamp}.geojson`

The augmented export combines:
1. Manual labels (`source: "manual"`) from search mode
2. Detection review labels (`source: "detection_review"`) from detection mode

---

## 2. Detection Results (Classifier Output)

Produced by the classification pipeline, consumed by detection mode for review.

### Structure

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-86.5, 32.3], [-86.4, 32.3], [-86.4, 32.4], [-86.5, 32.4], [-86.5, 32.3]]]
      },
      "properties": {
        "tile_id": "abc_123_456",
        "probability": 0.87,
        "class": "aquaculture"
      }
    }
  ]
}
```

### Properties Reference

| Property | Type | Values | Description |
|----------|------|--------|-------------|
| `tile_id` | string | Required | Unique identifier matching database |
| `probability` | float | 0.0 - 1.0 | Classifier confidence score |
| `class` | string | Any | Target class being detected |

### Loading Behavior

When loaded via `DatasetManager.load_from_content()`:
1. Triggers `state.enter_detection_mode(geojson)`
2. Sets `state.detection_mode = True`
3. Renders polygons on map with probability-based coloring
4. Enables tile review workflow in tile panel

---

## 3. DuckDB Schema

The `geo_embeddings` table stores tile metadata and embedding vectors.

### Table Definition

```sql
CREATE TABLE geo_embeddings (
    id BIGINT PRIMARY KEY,
    tile_id VARCHAR,                    -- Optional: original identifier
    embedding FLOAT[768],               -- or FLOAT[1024], UTINYINT[768] for INT8
    geometry GEOMETRY                   -- Point centroid of tile
);

-- Spatial index for fast nearest-point queries
CREATE INDEX geo_embeddings_geom_idx ON geo_embeddings USING RTREE(geometry);
```

### Column Details

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | BIGINT | No | Auto-incrementing primary key |
| `tile_id` | VARCHAR | Yes | Source tile identifier (e.g., MGRS grid cell) |
| `embedding` | FLOAT[N] | Yes | Foundation model embedding vector |
| `geometry` | GEOMETRY | No | Point at tile centroid |

### Embedding Types

| Type | Size | Use Case |
|------|------|----------|
| `FLOAT[768]` | 3KB/row | Standard 768-dim embeddings |
| `FLOAT[1024]` | 4KB/row | Larger models (e.g., Clay) |
| `UTINYINT[768]` | 768B/row | INT8 quantized (4x compression) |

### Common Queries

**Nearest point lookup** (for map clicks):
```sql
SELECT id, ST_AsText(geometry) AS wkt, embedding,
       ST_Distance(geometry, ST_Point(?, ?)) AS dist_m
FROM geo_embeddings
ORDER BY dist_m
LIMIT 1
```

**Fetch labeled embeddings**:
```sql
SELECT id, tile_id, ST_AsGeoJSON(geometry) AS geometry_json, embedding
FROM geo_embeddings
WHERE id IN (?, ?, ...)
```

**Similarity search** (after FAISS identifies candidates):
```sql
SELECT id, ST_AsGeoJSON(geometry) AS geometry_json, distance
FROM (
    SELECT id, geometry,
           embedding <-> CAST(? AS FLOAT[768]) AS distance
    FROM geo_embeddings
    WHERE embedding IS NOT NULL
    ORDER BY distance
    LIMIT ?
) sub
```

---

## 4. FAISS Index

Binary index file co-located with DuckDB database.

### File Naming Convention

```
{model_name}_{tile_pixels}_{overlap}_{resolution}.faiss
```

Example: `ssl4eo_32_16_10.faiss` → 32px tiles, 16px overlap, 10m resolution

### Index Types

| Type | Memory | Use Case |
|------|--------|----------|
| `IndexIVFPQ` | Low | Float embeddings with Product Quantization |
| `IndexIVFScalarQuantizer` | Low | INT8 embeddings with Scalar Quantization |
| `IndexFlatL2` | High | Small datasets, exact search |

### Search Parameters

```python
index.nprobe = 4096  # Number of clusters to search
```

Higher `nprobe` = more accurate but slower.

---

## 5. Tile Specification

Inferred from database filename, stored in `DataManager.tile_spec`:

```python
tile_spec = {
    "tile_size_px": 32,      # Tile dimension in pixels
    "meters_per_pixel": 10,  # Ground sample distance
}

# Derived: ground_size = tile_size_px * meters_per_pixel = 320m
```

### Naming Pattern

Database filename: `{name}_{pixels}_{overlap}_{resolution}.duckdb`

| Component | Example | Description |
|-----------|---------|-------------|
| `name` | `ssl4eo` | Model/dataset name |
| `pixels` | `32` | Tile size in pixels |
| `overlap` | `16` | Overlap between tiles in pixels |
| `resolution` | `10` | Meters per pixel |

---

## Related Files

- `geovibes/ui/datasets.py` — Save/load logic, format detection
- `geovibes/ui/state.py` — AppState with detection_data storage
- `geovibes/ui_config/constants.py` — DatabaseConstants, queries
- `geovibes/database/faiss_db.py` — FAISS index building
