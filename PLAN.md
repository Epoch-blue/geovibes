# Classification Layer Plan for GeoVibes

## Overview

Add a classification layer on top of GeoVibes that trains an XGBoost model on labeled embeddings and deploys it across the entire DuckDB index to generate a detection map.

## Input Specification

**Input GeoJSON format:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [...] },
      "properties": {
        "tile_id": "18SJH_32_16_10_5_12",  // Maps to DuckDB source_id
        "label": 1,                         // 1 = class of interest, 0 = negative
        "class": "forest"                   // Landcover class for stratification
      }
    }
  ]
}
```

## Architecture

### New Module Structure

```
geovibes/
├── classification/
│   ├── __init__.py           # Exports main classes
│   ├── classifier.py         # EmbeddingClassifier - XGBoost training/inference
│   ├── data_loader.py        # ClassificationDataLoader - GeoJSON parsing + embedding fetch
│   ├── inference.py          # BatchInference - memory-efficient scoring over DuckDB
│   ├── output.py             # OutputGenerator - tile geometry generation + GeoJSON export
│   └── pipeline.py           # ClassificationPipeline - orchestrates the full workflow
```

### Component Design

#### 1. ClassificationDataLoader (`data_loader.py`)

**Responsibilities:**
- Parse input GeoJSON with `tile_id`, `label`, `class` columns
- Match `tile_id` to DuckDB `source_id` column
- Fetch corresponding embeddings from DuckDB
- Stratified train/test split (equal negatives per class)

**Key Methods:**
```python
class ClassificationDataLoader:
    def __init__(self, duckdb_connection, geojson_path: str):
        self.conn = duckdb_connection
        self.geojson_path = geojson_path

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, test_df) with columns:
        - tile_id: str
        - embedding: np.ndarray
        - label: int (0 or 1)
        - class: str (landcover class)
        """
        pass

    def stratified_split(
        self,
        df: pd.DataFrame,
        test_fraction: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split with equal negatives per class.
        All positives go through standard stratified split.
        Negatives are sampled equally from each class.
        """
        pass
```

**Memory Considerations:**
- Fetch embeddings in chunks of 10,000 (existing pattern)
- Training data expected to be small (<100K samples), fits in memory

#### 2. EmbeddingClassifier (`classifier.py`)

**Responsibilities:**
- Train XGBoost binary classifier on embeddings
- Evaluate on test set with metrics (accuracy, precision, recall, F1, AUC)
- Save/load trained model
- Predict probabilities in batch mode

**Key Methods:**
```python
class EmbeddingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,  # Use all cores
        tree_method: str = "hist",  # Memory-efficient for M1
    ):
        self.model = XGBClassifier(...)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train model, optionally with early stopping on validation set.
        Returns training metrics.
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Returns dict with accuracy, precision, recall, f1, auc_roc.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of positive class."""
        pass

    def save(self, path: str) -> None:
        """Save model to JSON format (XGBoost native)."""
        pass

    @classmethod
    def load(cls, path: str) -> "EmbeddingClassifier":
        """Load model from JSON file."""
        pass
```

**XGBoost Configuration for M1 Mac:**
- `tree_method="hist"`: Histogram-based, memory-efficient
- `device="cpu"`: M1 doesn't support XGBoost GPU
- `n_jobs=-1`: Use all CPU cores
- `max_bin=256`: Default, good for embeddings

#### 3. BatchInference (`inference.py`)

**Responsibilities:**
- Score all embeddings in DuckDB in batches
- Track which IDs exceed probability threshold
- Memory-efficient streaming through the database

**Key Methods:**
```python
class BatchInference:
    def __init__(
        self,
        classifier: EmbeddingClassifier,
        duckdb_connection,
        batch_size: int = 50_000,  # ~200MB per batch for 1024-dim embeddings
    ):
        self.classifier = classifier
        self.conn = duckdb_connection
        self.batch_size = batch_size

    def run(
        self,
        probability_threshold: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[int]:
        """
        Returns list of DuckDB IDs where P(positive) >= threshold.

        Memory budget per batch (1024-dim float32):
        - 50K embeddings × 1024 dims × 4 bytes = ~200MB
        - Plus XGBoost internal buffers ~100MB
        - Total ~300MB per batch, safe for 32GB system
        """
        pass

    def _iterate_embeddings(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator yielding (ids, embeddings) batches.
        Uses DuckDB cursor with LIMIT/OFFSET for memory efficiency.
        """
        pass
```

**Batch Size Calculation:**
- 1M embeddings × 1024 dims × 4 bytes = 4GB (full dataset)
- Batch of 50K = 200MB embeddings + ~100MB XGBoost overhead
- Safe margin for 32GB RAM with other processes

#### 4. OutputGenerator (`output.py`)

**Responsibilities:**
- Fetch metadata (geometry, source_id) for detected IDs
- Generate tile geometries from centroids (UTM projection)
- Union overlapping tiles
- Export final GeoJSON

**Key Methods:**
```python
class OutputGenerator:
    def __init__(
        self,
        duckdb_connection,
        tile_size_px: int = 32,
        tile_overlap_px: int = 16,
        resolution_m: float = 10.0,
    ):
        self.conn = duckdb_connection
        self.tile_size_m = tile_size_px * resolution_m  # 320m
        self.overlap_m = tile_overlap_px * resolution_m  # 160m

    def fetch_detection_metadata(
        self,
        detection_ids: List[int],
        chunk_size: int = 10_000,
    ) -> gpd.GeoDataFrame:
        """
        Fetch id, source_id, geometry, probability for detected IDs.
        Returns GeoDataFrame with point geometries.
        """
        pass

    def generate_tile_geometries(
        self,
        detections_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Convert point centroids to square tile polygons.
        Projects to appropriate UTM zone, buffers, reprojects to WGS84.
        """
        pass

    def union_tiles(
        self,
        tiles_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Union overlapping tiles into larger polygons.
        Uses unary_union followed by explode for multi-polygons.
        """
        pass

    def export_geojson(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
        include_individual_tiles: bool = True,
    ) -> Dict[str, str]:
        """
        Export two files:
        - {name}_detections.geojson: Individual tile polygons with scores
        - {name}_union.geojson: Unioned detection regions

        Returns dict with paths to both files.
        """
        pass
```

**UTM Projection Strategy:**
- Parse MGRS zone from `source_id` (e.g., "18SJH" → UTM zone 18N)
- Group detections by UTM zone for batch processing
- Buffer points by `tile_size_m / 2` with square cap (`cap_style=3`)
- Reproject back to WGS84 (EPSG:4326) for output

#### 5. ClassificationPipeline (`pipeline.py`)

**Responsibilities:**
- Orchestrate the full workflow
- Provide progress reporting
- Handle errors gracefully

**Key Methods:**
```python
class ClassificationPipeline:
    def __init__(
        self,
        duckdb_path: str,
        faiss_path: Optional[str] = None,  # Not needed for classification
    ):
        self.duckdb_path = duckdb_path
        self.conn = None

    def run(
        self,
        geojson_path: str,
        output_dir: str,
        test_fraction: float = 0.2,
        probability_threshold: float = 0.5,
        xgb_params: Optional[Dict] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Load and split training data
        2. Train XGBoost classifier
        3. Evaluate on test set
        4. Run inference on all embeddings
        5. Generate output GeoJSON

        Returns:
        {
            "metrics": {"accuracy": 0.95, "f1": 0.87, ...},
            "num_detections": 12345,
            "output_files": {
                "detections": "output/detections.geojson",
                "union": "output/union.geojson",
                "model": "output/model.json",
            }
        }
        """
        pass

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        self._disconnect()
```

## Memory Budget Analysis

**Target: 32GB RAM M1 Mac**

| Component | Peak Memory | Notes |
|-----------|-------------|-------|
| DuckDB connection | ~1GB | Process memory limit set to 8GB |
| Training data (100K × 1024) | ~400MB | Float32 embeddings |
| XGBoost training | ~2GB | Histogram method is efficient |
| Inference batch (50K × 1024) | ~200MB | Per batch |
| XGBoost inference buffers | ~100MB | Per batch |
| Detection IDs list | ~80MB | 10M IDs × 8 bytes |
| Output GeoDataFrame | ~500MB | Geometries + metadata |
| **Total Peak** | **~5GB** | Safe margin |

**DuckDB Memory Settings (reduced for coexistence):**
```python
SET memory_limit='8GB';
SET max_memory='8GB';
SET temp_directory='/tmp';
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Classification Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Input      │    │   DuckDB     │    │   Training   │       │
│  │   GeoJSON    │───▶│   Lookup     │───▶│   Data       │       │
│  │ (tile_id,    │    │ (source_id   │    │ (embeddings  │       │
│  │  label,      │    │  matching)   │    │  + labels)   │       │
│  │  class)      │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐       │
│                                          │  Stratified  │       │
│                                          │  Train/Test  │       │
│                                          │  Split       │       │
│                                          │ (equal neg/  │       │
│                                          │  class)      │       │
│                                          └──────────────┘       │
│                                                  │               │
│                      ┌───────────────────────────┼───────┐       │
│                      ▼                           ▼       │       │
│               ┌──────────────┐           ┌──────────────┐│       │
│               │   XGBoost    │           │   Test Set   ││       │
│               │   Training   │           │   Evaluation ││       │
│               │   (hist      │──────────▶│   (metrics)  ││       │
│               │   method)    │           │              ││       │
│               └──────────────┘           └──────────────┘│       │
│                      │                                   │       │
│                      ▼                                   │       │
│               ┌──────────────┐                           │       │
│               │   Batch      │                           │       │
│               │   Inference  │◀──────────────────────────┘       │
│               │   (50K/batch)│                                   │
│               └──────────────┘                                   │
│                      │                                           │
│                      ▼                                           │
│               ┌──────────────┐                                   │
│               │   Threshold  │                                   │
│               │   Filtering  │                                   │
│               │   (P >= t)   │                                   │
│               └──────────────┘                                   │
│                      │                                           │
│                      ▼                                           │
│               ┌──────────────┐                                   │
│               │   Fetch      │                                   │
│               │   Geometries │                                   │
│               │   (detected  │                                   │
│               │   IDs)       │                                   │
│               └──────────────┘                                   │
│                      │                                           │
│                      ▼                                           │
│               ┌──────────────┐                                   │
│               │   Generate   │                                   │
│               │   Tile       │                                   │
│               │   Polygons   │                                   │
│               │   (UTM)      │                                   │
│               └──────────────┘                                   │
│                      │                                           │
│                      ▼                                           │
│               ┌──────────────┐    ┌──────────────┐               │
│               │   Union      │───▶│   Output     │               │
│               │   Tiles      │    │   GeoJSON    │               │
│               └──────────────┘    └──────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Phase 1: Data Loading (data_loader.py)
- [ ] Create ClassificationDataLoader class
- [ ] Implement GeoJSON parsing with tile_id, label, class extraction
- [ ] Implement source_id matching query to DuckDB
- [ ] Implement stratified split with equal negatives per class
- [ ] Write unit tests for data loading

### Phase 2: Classifier (classifier.py)
- [ ] Create EmbeddingClassifier class with XGBoost
- [ ] Implement fit() with optional early stopping
- [ ] Implement evaluate() with standard metrics
- [ ] Implement predict_proba() for batch inference
- [ ] Implement save/load for model persistence
- [ ] Write unit tests for classifier

### Phase 3: Batch Inference (inference.py)
- [ ] Create BatchInference class
- [ ] Implement batched iteration over DuckDB embeddings
- [ ] Implement threshold filtering
- [ ] Add progress callback support
- [ ] Write unit tests for inference

### Phase 4: Output Generation (output.py)
- [ ] Create OutputGenerator class
- [ ] Implement detection metadata fetching
- [ ] Implement tile geometry generation (UTM projection)
- [ ] Implement tile union operation
- [ ] Implement GeoJSON export
- [ ] Write unit tests for output generation

### Phase 5: Pipeline Integration (pipeline.py)
- [ ] Create ClassificationPipeline class
- [ ] Implement run() orchestration method
- [ ] Add context manager support
- [ ] Add comprehensive progress reporting
- [ ] Write integration tests

### Phase 6: Testing & Documentation
- [ ] Create example notebook demonstrating the workflow
- [ ] Add CLI interface for pipeline execution
- [ ] Update CLAUDE.md with classification documentation

## Test Strategy (TDD)

For each module:
1. Create test file `tests/test_classification_{module}.py`
2. Write test cases for expected behavior
3. Create stub implementation that returns expected types
4. Run tests (expect failures)
5. Implement actual logic
6. Run tests (expect passes)

**Test fixtures:**
- Small mock DuckDB with ~100 embeddings
- Sample GeoJSON with mixed classes
- Pre-computed expected metrics

## Dependencies

**New dependencies to add to pyproject.toml:**
```toml
dependencies = [
    # ... existing ...
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",  # For metrics and stratified split
]
```

## Example Usage

```python
from geovibes.classification import ClassificationPipeline

# Run full pipeline
with ClassificationPipeline(duckdb_path="local_databases/model.duckdb") as pipeline:
    results = pipeline.run(
        geojson_path="training_data/labeled_tiles.geojson",
        output_dir="classification_output/",
        test_fraction=0.2,
        probability_threshold=0.7,
        xgb_params={"n_estimators": 200, "max_depth": 8},
    )

print(f"Test F1: {results['metrics']['f1']:.3f}")
print(f"Detections: {results['num_detections']}")
print(f"Output: {results['output_files']['union']}")
```

## Open Questions (Resolved)

1. **ID matching**: Use `tile_id` → `source_id` matching ✓
2. **Stratification**: Equal negatives per class ✓
3. **Output format**: GeoJSON ✓
4. **Scale**: 1-10M embeddings, batching required ✓
