# UI Widgets

GeoVibes uses a hybrid approach combining **ipyvuetify** (Material Design components) for the side panel with **ipywidgets** for the tile panel and map controls.

---

## Widget Library Strategy

| Component | Library | Rationale |
|-----------|---------|-----------|
| Side panel | ipyvuetify | Polished Material Design buttons, toggles, selects |
| Tile panel | ipywidgets | Lightweight cards for many tiles |
| Map | ipyleaflet | Interactive map with drawing controls |

---

## Side Panel Widget Hierarchy

```
VBox (geovibes-panel)
├── HTML (CSS injection)
├── v.Card (search_card)
│   ├── v.Row
│   │   ├── v.Col(10) → v.Btn (search_btn)
│   │   └── v.Col(2) → v.Btn (tiles_button)
│   └── v.Row (slider_row)
│       ├── v.Col(10) → v.Slider (neighbors_slider)
│       └── v.Col(2) → v.Html (neighbors_label)
├── v.Card (label_card)
│   ├── v.Html (section-label "LABEL")
│   └── v.BtnToggle (label_toggle)
│       ├── v.Btn [mdi-thumb-up-outline]
│       ├── v.Btn [mdi-thumb-down-outline]
│       └── v.Btn [mdi-eraser]
├── v.Card (mode_card)
│   ├── v.Html (section-label "MODE")
│   └── v.BtnToggle (selection_mode)
│       ├── v.Btn "• Point"
│       └── v.Btn "▢ Polygon"
├── v.Card (detection_controls) [hidden by default]
│   ├── v.Html (section-label "DETECTION THRESHOLD")
│   ├── HBox
│   │   ├── FloatSlider (detection_threshold_slider)
│   │   └── FloatText (detection_threshold_text)
│   └── Label (detection_status_label)
├── VBox (accordion_container)
│   ├── v.Card (database_card) [if databases exist]
│   │   ├── v.Html (section-label "DATABASE")
│   │   └── v.Select (database_dropdown)
│   ├── v.Card (basemaps_card)
│   │   ├── v.Html (section-label "BASEMAP")
│   │   └── v.Select (basemap_dropdown)
│   └── v.Card (export_card)
│       ├── v.Html (section-label "EXPORT & TOOLS")
│       ├── v.BtnToggle
│       │   ├── v.Btn (save_btn) [mdi-content-save-outline]
│       │   └── v.Btn (load_btn) [mdi-folder-open-outline]
│       └── v.BtnToggle
│           ├── v.Btn (add_vector_btn) [mdi-vector-polygon]
│           └── v.Btn (google_maps_btn) [mdi-google-maps]
├── v.Btn (reset_btn) [mdi-trash-can-outline]
└── VBox (hidden_uploads)
    ├── FileUpload (file_upload)
    └── FileUpload (vector_file_upload)
```

---

## ipyvuetify Event Patterns

### v.Btn (Button)

```python
# Click handler
btn.on_event("click", lambda *args: do_something())

# With icon
v.Btn(
    small=True,
    children=[
        v.Icon(small=True, children=["mdi-magnify"]),
        "Search",
    ],
)
```

### v.BtnToggle (Toggle Group)

```python
# Index-based selection
toggle = v.BtnToggle(
    v_model=0,           # Selected index
    mandatory=True,      # Always one selected
    children=[
        v.Btn(small=True, children=["Option 1"]),
        v.Btn(small=True, children=["Option 2"]),
    ],
)

# Event handler receives index
toggle.observe(handler, names="v_model")

def handler(change):
    idx = change["new"]  # 0, 1, 2, ...
    value = options[idx]
```

### v.Select (Dropdown)

```python
dropdown = v.Select(
    v_model="value",
    items=[
        {"text": "Display Name", "value": "actual_value"},
    ],
    dense=True,
    outlined=True,
    hide_details=True,
)

# Event handler receives value
dropdown.observe(handler, names="v_model")

def handler(change):
    value = change["new"]  # The selected value string
```

### v.Slider

```python
slider = v.Slider(
    v_model=1000,
    min=100,
    max=25000,
    step=100,
    thumb_label=True,
    hide_details=True,
)

slider.observe(handler, names="v_model")
```

---

## Icon Systems

**Critical**: ipyvuetify uses Material Design Icons (MDI), not FontAwesome.

| Library | Icon System | Example |
|---------|------------|---------|
| ipyvuetify | MDI | `mdi-thumb-up-outline` |
| ipywidgets Button | FontAwesome | `icon="fa-thumbs-up"` |

MDI icon reference: https://materialdesignicons.com/

Common icons used:
- `mdi-magnify` — Search
- `mdi-thumb-up-outline` — Positive label
- `mdi-thumb-down-outline` — Negative label
- `mdi-eraser` — Erase
- `mdi-content-save-outline` — Save
- `mdi-folder-open-outline` — Load
- `mdi-vector-polygon` — Vector layer
- `mdi-google-maps` — Google Maps
- `mdi-trash-can-outline` — Reset
- `mdi-view-grid-outline` — Tile panel

---

## Tile Panel Widget Hierarchy

```
VBox (TilePanel)
├── HTML (TILE_PANEL_CSS)
├── HBox (header)
│   ├── Dropdown (sort_dropdown) ["Similar", "Dissimilar"]
│   ├── Dropdown (basemap_dropdown) ["HUTCH_TILE", "MAPTILER", "GOOGLE_HYBRID"]
│   └── Button (close_btn)
├── GridBox (tile_grid)
│   └── [VBox (tile_card)] × N
│       ├── Image (tile_image)
│       ├── HBox (info_row)
│       │   ├── Label (rank "#1")
│       │   └── Label (distance/probability)
│       └── HBox (button_row)
│           ├── Button (pos_btn) 👍
│           └── Button (neg_btn) 👎
└── HBox (footer)
    ├── Label (page_info)
    └── Button (load_more_btn)
```

---

## Tile Card States

CSS classes applied based on label state:

| State | CSS Class | Border Color |
|-------|-----------|--------------|
| Unlabeled | — | transparent |
| Positive | `tile-positive` | `#22c55e` (green) |
| Negative | `tile-negative` | `#ef4444` (red) |

Applied in `tiles.py:_build_tile_widget()`:
```python
if point_id in self.state.pos_ids:
    card.add_class("tile-positive")
elif point_id in self.state.neg_ids:
    card.add_class("tile-negative")
```

---

## Map Layer Stack

```
ipyleaflet.Map
├── TileLayer (basemap_layer)
├── GeoJSON (boundary_layer) [region outline]
├── GeoJSON (search_layer) [search results]
├── GeoJSON (pos_layer) [positive labels]
├── GeoJSON (neg_layer) [negative labels]
├── GeoJSON (erase_layer) [erase feedback]
├── GeoJSON (detection_layer) [detection polygons]
├── GeoJSON (vector_layer) [user overlays]
├── GeoJSON (highlight_layer) [tile highlight]
└── DrawControl (draw_control) [polygon drawing]
```

Layer management in `map_manager.py`:
```python
def update_search_layer(geojson, style_callback=None):
    # Remove old, add new

def update_label_layers(pos_geojson, neg_geojson, erase_geojson):
    # Update all label layers

def update_detection_layer(geojson, style_callback=None):
    # For detection mode
```

---

## CSS Injection

Custom CSS is injected via `HTML` widget at the top of containers:

```python
SIDE_PANEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.geovibes-panel,
.geovibes-panel * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
/* ... more styles ... */
</style>
"""

css_widget = HTML(SIDE_PANEL_CSS)
panel = VBox([css_widget, ...])
panel.add_class("geovibes-panel")
```

---

## State-Widget Bindings

| Widget | State Variable | Type |
|--------|----------------|------|
| `label_toggle.v_model` | `state.current_label` | Index → "Positive"/"Negative"/"Erase" |
| `selection_mode.v_model` | `state.selection_mode` | Index → "point"/"polygon" |
| `neighbors_slider.v_model` | — | Direct read on search |
| `basemap_dropdown.v_model` | `state.tile_basemap` | Basemap name string |
| `database_dropdown.v_model` | `data.current_database_path` | DB path string |
| `detection_threshold_slider.value` | — | Direct read for filtering |

---

## Async Tile Loading

Tile images are loaded asynchronously using `ThreadPoolExecutor`:

```python
# In TilePanel
self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def _load_tile_images(self, tiles):
    futures = {
        self._executor.submit(get_map_image, lat, lon, zoom, url): tile_id
        for tile_id, (lat, lon) in tiles
    }
    for future in concurrent.futures.as_completed(futures):
        tile_id = futures[future]
        image_bytes = future.result()
        # Update widget in UI thread
        asyncio.get_event_loop().call_soon_threadsafe(
            self._update_tile_image, tile_id, image_bytes
        )
```

---

## Related Files

- `geovibes/ui/app.py` — Side panel construction (`_build_side_panel`)
- `geovibes/ui/tiles.py` — Tile panel (`TilePanel` class)
- `geovibes/ui/map_manager.py` — Map and layer management
- `geovibes/ui_config/constants.py` — Colors, dimensions, styles
