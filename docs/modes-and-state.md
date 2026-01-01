# Modes and State Management

GeoVibes operates in two distinct modes that fundamentally change how the application behaves. Understanding these modes is critical for development.

## Overview

| Mode | Purpose | Entry Point | Data Source |
|------|---------|-------------|-------------|
| **Search Mode** | Build training data via similarity search | Default on startup | FAISS index + user labels |
| **Detection Mode** | Review pre-computed classifier predictions | Load detection GeoJSON | Classification pipeline output |

---

## State Variables

All state is managed in `AppState` (`geovibes/ui/state.py`):

```python
@dataclass
class AppState:
    # Search mode state
    pos_ids: List[str]                    # Point IDs labeled positive
    neg_ids: List[str]                    # Point IDs labeled negative
    cached_embeddings: Dict[str, np.ndarray]  # id → embedding vector
    query_vector: Optional[np.ndarray]    # Computed search vector

    # Selection state
    selection_mode: str = "point"         # "point" or "polygon"
    current_label: str = "Positive"       # "Positive", "Negative", "Erase"

    # Detection mode state
    detection_mode: bool = False          # True when reviewing detections
    detection_data: Optional[Dict]        # Loaded GeoJSON with probabilities
    detection_labels: Dict[str, int]      # tile_id → 1 (confirm) or 0 (reject)

    # Results
    last_search_results_df: Optional[pd.DataFrame]
    detections_with_embeddings: Optional[gpd.GeoDataFrame]
```

---

## Search Mode

### Purpose
Build a training dataset by iteratively labeling points and finding similar locations.

### Flow

```
User clicks map
       ↓
_on_map_interaction()
       ↓
label_point(lon, lat)
       ↓
├── Query DuckDB for nearest embedding
├── Cache embedding in state.cached_embeddings
├── Apply label to state.pos_ids or state.neg_ids
├── _update_layers() → refresh map markers
└── _update_query_vector() → recompute search vector
       ↓
User clicks "Search"
       ↓
search_click()
       ↓
_search_faiss()
       ↓
├── query_vector.reshape(1, -1) → FAISS search
├── Filter out already-labeled IDs
└── _process_search_results() → update map + tile panel
```

### Query Vector Formula

```python
# In AppState.update_query_vector()
pos_vec = np.mean([embeddings for pos_ids], axis=0)
neg_vec = np.mean([embeddings for neg_ids], axis=0)  # zeros if empty
query_vector = 2 * pos_vec - neg_vec
```

This formula pushes the query away from negatives while centering on positives.

### Tile Sorting (Search Mode)

Results from FAISS are sorted by **distance** (L2):
- **Similar** button: Keep order (low distance = most similar first)
- **Dissimilar** button: Reverse order (high distance first)

```python
# In tiles.py _render_current_page()
is_detection = getattr(self.state, "detection_mode", False)
should_reverse = (self._sort_order == "Dissimilar") != is_detection
```

---

## Detection Mode

### Purpose
Review predictions from the classification pipeline. Confirm or reject detections to create correction datasets.

### Entry

```python
# In AppState
def enter_detection_mode(self, detection_geojson: Dict) -> None:
    self.detection_mode = True
    self.detection_data = detection_geojson
    self.detection_labels.clear()
```

Detection GeoJSON format (from classification pipeline):
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [...]},
      "properties": {
        "tile_id": "abc123",
        "probability": 0.87,
        "class": "aquaculture"
      }
    }
  ]
}
```

### Flow

```
Load detection GeoJSON
       ↓
state.enter_detection_mode(geojson)
       ↓
User clicks on detection polygon
       ↓
_on_map_interaction()
       ↓
_handle_detection_click(lon, lat)  # Different from label_point!
       ↓
├── Find clicked feature by geometry.contains(point)
├── state.label_detection(tile_id, 1 or 0)
└── Update tile styling to show confirmed/rejected
```

### Tile Sorting (Detection Mode)

Results are sorted by **probability**:
- **Similar** button: Reverse order (high probability first) — "most confident"
- **Dissimilar** button: Keep order (low probability first) — "least confident"

This is **inverted** from search mode because:
- Search results arrive sorted by distance ascending (low = similar)
- Detection results arrive sorted by probability ascending (low = uncertain)

### Labeling in Detection Mode

```python
# In tiles.py _build_tile_widget()
if self.state.detection_mode:
    detection_label = self.state.detection_labels.get(point_id)
    if detection_label == 1:   # Confirmed positive
        # Green styling
    if detection_label == 0:   # Rejected (false positive)
        # Red styling
```

### Saving Detection Reviews

When saving a dataset from detection mode, labels are exported as:
```json
{
  "properties": {
    "label": 1,  // or -1 for rejected
    "class": "aquaculture",
    "source": "detection_review"  // Distinguishes from manual labels
  }
}
```

---

## Mode Transitions

### Search → Detection

1. User loads a detection GeoJSON file
2. `state.enter_detection_mode(geojson)` is called
3. `state.detection_mode = True`
4. Map click behavior changes to `_handle_detection_click()`
5. Tile panel shows probability-sorted results

### Detection → Search

1. User clicks "Exit Detection Mode" or loads a new database
2. `state.exit_detection_mode()` is called
3. `state.detection_mode = False`
4. `state.detection_data = None`
5. `state.detection_labels.clear()`
6. Map click behavior reverts to `label_point()`

### Reset (Either Mode)

```python
def reset(self) -> None:
    self.pos_ids.clear()
    self.neg_ids.clear()
    self.cached_embeddings.clear()
    self.query_vector = None
    self.detection_mode = False
    self.detection_data = None
    self.detection_labels.clear()
    # ... other state reset
```

---

## Key Code Paths by Mode

| Action | Search Mode | Detection Mode |
|--------|-------------|----------------|
| Map click | `label_point()` | `_handle_detection_click()` |
| Label storage | `state.pos_ids` / `state.neg_ids` | `state.detection_labels` |
| Tile sort | Distance (low=similar) | Probability (low=uncertain) |
| Save format | `label: 1/-1` | `label: 1/-1, source: detection_review` |

---

## Common Development Pitfalls

### 1. Forgetting to check `detection_mode`

Many methods behave differently based on mode. Always check:
```python
if self.state.detection_mode:
    # Detection-specific behavior
else:
    # Search-specific behavior
```

### 2. Sort order confusion

The `should_reverse` logic in `tiles.py` is intentionally inverted for detection mode. Don't "fix" this — it's correct.

### 3. State leakage between modes

When transitioning modes, ensure relevant state is cleared:
- `detection_labels` should be empty in search mode
- `query_vector` may be stale after entering detection mode

### 4. Label semantics differ

- Search mode: `pos_ids`/`neg_ids` are **training labels** for building query
- Detection mode: `detection_labels` are **verification labels** (confirm/reject predictions)

---

## Related Files

- `geovibes/ui/state.py` — AppState dataclass
- `geovibes/ui/app.py` — Mode-specific click handlers
- `geovibes/ui/tiles.py` — Sort order logic, tile styling
- `geovibes/ui/datasets.py` — Save/load with mode awareness
