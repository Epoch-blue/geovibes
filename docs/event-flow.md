# Event Flow

This document traces the method chains triggered by user interactions in the GeoVibes UI.

---

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Action   │────▶│  Event Handler   │────▶│   State Update  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                         │
                                │                         ▼
                                │                ┌─────────────────┐
                                └───────────────▶│  UI Refresh     │
                                                 └─────────────────┘
```

All events are wired in `GeoVibes._wire_events()` (app.py:584).

---

## Map Click (Search Mode)

**Trigger**: User clicks on map in point selection mode

```
_on_map_interaction(**kwargs)                    # app.py:816
    │
    ├─▶ _update_status(lat, lon)                 # Update coordinates in status bar
    │
    ├─▶ [If Ctrl+Click] → Open Google Maps       # Bail out early
    │
    ├─▶ [If detection_mode] → _handle_detection_click()
    │
    └─▶ [If point mode] → label_point(lon, lat)  # app.py:846
            │
            ├─▶ data.nearest_point(lon, lat)     # Query DuckDB for closest embedding
            │       Returns: (id, wkt, distance, embedding)
            │
            ├─▶ state.cached_embeddings[id] = embedding
            │
            ├─▶ [If Erase mode]
            │       └─▶ state.remove_label(id)
            │       └─▶ map_manager.update_label_layers(erase_geojson)
            │
            ├─▶ [Else] state.apply_label(id, select_val)
            │       └─▶ toggle_label() in AppState
            │
            ├─▶ _update_layers()                 # app.py:1183
            │       └─▶ _geojson_for_ids(pos_ids)
            │       └─▶ _geojson_for_ids(neg_ids)
            │       └─▶ map_manager.update_label_layers(pos, neg, erase)
            │
            └─▶ _update_query_vector()           # app.py:1213
                    └─▶ _fetch_embeddings(pos_ids)
                    └─▶ _fetch_embeddings(neg_ids)
                    └─▶ state.update_query_vector()
                            │
                            └─▶ query = 2 * mean(pos) - mean(neg)
```

---

## Map Click (Detection Mode)

**Trigger**: User clicks on detection polygon

```
_on_map_interaction(**kwargs)                    # app.py:816
    │
    └─▶ _handle_detection_click(lon, lat)        # app.py:895
            │
            ├─▶ For each feature in detection_data:
            │       geom = shapely.shape(feature.geometry)
            │       if geom.contains(Point(lon, lat)):
            │           tile_id = props.tile_id
            │           ...
            │
            ├─▶ [If Positive mode]
            │       └─▶ state.label_detection(tile_id, 1)
            │
            ├─▶ [If Negative mode]
            │       └─▶ state.label_detection(tile_id, 0)
            │
            ├─▶ [If Erase mode]
            │       └─▶ del state.detection_labels[tile_id]
            │
            └─▶ _refresh_detection_layer()       # app.py:1354
                    └─▶ _filter_detection_layer(threshold)
                            └─▶ map_manager.update_detection_layer(filtered)
```

---

## Search Button Click

**Trigger**: User clicks "Search" button

```
search_btn.on_event("click")                     # app.py:586
    │
    └─▶ search_click(None)                       # app.py:1046
            │
            ├─▶ state.tile_page = 0
            ├─▶ _reset_tiles_button()
            │
            ├─▶ [If no query_vector] → Show warning, return
            │
            └─▶ _search_faiss()                  # app.py:1056
                    │
                    ├─▶ n_neighbors = neighbors_slider.v_model
                    │
                    ├─▶ query_vector.reshape(1, -1).astype('float32')
                    │
                    ├─▶ data.faiss_index.search(query, n, params=IVF(nprobe=4096))
                    │       Returns: distances[], ids[]
                    │
                    ├─▶ data.query_search_metadata(faiss_ids)
                    │       Returns: DataFrame with geometry_wkt, geometry_json
                    │
                    └─▶ _process_search_results(df, n_neighbors)  # app.py:1093
                            │
                            ├─▶ Filter out already-labeled IDs
                            │
                            ├─▶ Build GeoDataFrame → state.detections_with_embeddings
                            │
                            ├─▶ Build GeoJSON with distance→color mapping
                            │       └─▶ UIConstants.distance_to_color()
                            │
                            ├─▶ state.last_search_results_df = filtered
                            │
                            ├─▶ map_manager.update_search_layer(geojson, style_callback)
                            │
                            └─▶ tile_panel.update_results(df, on_ready=_on_tiles_ready)
                                    │
                                    └─▶ Async tile image loading with ThreadPoolExecutor
```

---

## Label Toggle Change

**Trigger**: User clicks Positive/Negative/Erase button

```
label_toggle.observe(_on_label_toggle_change, names="v_model")
    │
    └─▶ _on_label_toggle_change(change)          # app.py:647
            │
            ├─▶ idx = change["new"]
            │       0 = Positive, 1 = Negative, 2 = Erase
            │
            ├─▶ value = _label_values[idx]
            │
            ├─▶ state.set_label_mode(value)      # state.py:41
            │       └─▶ state.current_label = value
            │       └─▶ state.select_val = UIConstants.{POS|NEG|ERASE}_LABEL
            │
            └─▶ _update_status()
```

---

## Selection Mode Toggle (Point/Polygon)

**Trigger**: User clicks Point/Polygon button

```
selection_mode.observe(_on_selection_mode_change, names="v_model")
    │
    └─▶ _on_selection_mode_change(change)        # app.py:654
            │
            ├─▶ idx = change["new"]
            │       0 = point, 1 = polygon
            │
            ├─▶ value = _mode_values[idx]
            │
            ├─▶ state.selection_mode = value
            ├─▶ state.lasso_mode = (value == "polygon")
            ├─▶ state.execute_label_point = (value != "polygon")
            │
            └─▶ _update_status()
```

---

## Polygon Draw

**Trigger**: User draws polygon in polygon mode

```
map_manager.register_draw_handler(_handle_draw)
    │
    └─▶ _handle_draw(target, action, geo_json)   # app.py:941
            │
            ├─▶ [If action == "created" and Polygon]:
            │       │
            │       ├─▶ polygon = shapely.Polygon(coords)
            │       │
            │       ├─▶ [If detection_mode]:
            │       │       └─▶ _label_detections_in_polygon(polygon)
            │       │               └─▶ For each feature:
            │       │                   if polygon.intersects(geom):
            │       │                       state.label_detection(tile_id, label)
            │       │               └─▶ _refresh_detection_layer()
            │       │
            │       ├─▶ [Else - Search mode]:
            │       │       └─▶ Query detections_with_embeddings.within(polygon)
            │       │       └─▶ Or fallback: ST_Within query on DuckDB
            │       │       └─▶ _fetch_embeddings(point_ids)
            │       │       └─▶ For each point: state.apply_label(pid, select_val)
            │       │       └─▶ _update_layers()
            │       │       └─▶ _update_query_vector()
            │       │
            │       └─▶ map_manager.draw_control.clear()
            │
            ├─▶ [If action == "drawstart"]:
            │       └─▶ state.polygon_drawing = True
            │
            └─▶ [If action == "deleted"]:
                    └─▶ state.polygon_drawing = False
```

---

## Tile Panel Label Click

**Trigger**: User clicks 👍/👎 button on a tile card

```
tile_widget.on_click()                           # tiles.py
    │
    └─▶ on_label(point_id, row, label)
            │
            └─▶ _handle_tile_label(point_id, row, label)  # app.py:1242
                    │
                    ├─▶ [If detection_mode]:
                    │       ├─▶ state.label_detection(tile_id, 1 or 0)
                    │       └─▶ _refresh_detection_layer()
                    │
                    └─▶ [Else]:
                            ├─▶ _fetch_embeddings([point_id])
                            ├─▶ state.apply_label(point_id, label)
                            ├─▶ _update_layers()
                            └─▶ _update_query_vector()
```

---

## Tile Panel Center Click

**Trigger**: User clicks tile image to center map

```
tile_image.on_click()                            # tiles.py
    │
    └─▶ on_center(row)
            │
            └─▶ _handle_tile_center(row)         # app.py:1281
                    │
                    ├─▶ geom = shapely.wkt.loads(row["geometry_wkt"])
                    ├─▶ map_manager.center_on(lat, lon, zoom=14)
                    ├─▶ polygon = _tile_polygon_from_spec(lat, lon)
                    │       └─▶ Convert to UTM, create box, transform back
                    └─▶ map_manager.highlight_polygon(polygon, color="red")
```

---

## File Upload (Dataset/Vector)

**Trigger**: User uploads a file via Load or Vector button

```
file_upload.observe(_on_file_upload, names="value")
    │
    └─▶ _on_file_upload(change)                  # app.py:743
            │
            ├─▶ content = DatasetManager.read_upload_content(file_info)
            │
            ├─▶ reset_all()
            │
            ├─▶ dataset_manager.load_from_content(content, filename)
            │       │
            │       ├─▶ detect_geojson_type(data)
            │       │       Returns: "labeled" | "detections" | "vector_layer"
            │       │
            │       ├─▶ [If labeled]:
            │       │       └─▶ _apply_geojson_payload()
            │       │               └─▶ state.reset()
            │       │               └─▶ For each feature:
            │       │                   state.pos_ids.append() or neg_ids.append()
            │       │                   state.cached_embeddings[id] = embedding
            │       │
            │       ├─▶ [If detections]:
            │       │       └─▶ _apply_detection_payload()
            │       │               └─▶ state.detection_mode = True
            │       │               └─▶ state.detection_data = payload
            │       │               └─▶ map_manager.update_detection_layer()
            │       │
            │       └─▶ [If vector_layer]:
            │               └─▶ map_manager.set_vector_layer()
            │
            ├─▶ [If detection_mode]:
            │       └─▶ Show detection_controls
            │       └─▶ Set slider min/max from probability range
            │       └─▶ _filter_detection_layer(min_prob)
            │       └─▶ _update_detection_tiles()
            │
            └─▶ [Else]:
                    └─▶ _update_layers()
                    └─▶ _update_query_vector()
```

---

## Database Switch

**Trigger**: User selects different database from dropdown

```
database_dropdown.observe(_on_database_change, names="v_model")
    │
    └─▶ _on_database_change(change)              # app.py:695
            │
            ├─▶ data.switch_database(new_path)
            │       └─▶ Close old connection
            │       └─▶ Open new DuckDB connection
            │       └─▶ Load new FAISS index
            │       └─▶ Update tile_spec, center, boundary
            │
            ├─▶ map_manager.center_on(center_y, center_x)
            ├─▶ map_manager.update_boundary_layer(boundary_path)
            │
            └─▶ reset_all()
```

---

## Detection Threshold Change

**Trigger**: User moves threshold slider

```
detection_threshold_slider.observe(_on_detection_threshold_change, names="value")
    │
    └─▶ _on_detection_threshold_change(change)   # app.py:720
            │
            ├─▶ Sync detection_threshold_text with slider
            │
            ├─▶ _filter_detection_layer(threshold)
            │       └─▶ Filter features where probability >= threshold
            │       └─▶ map_manager.update_detection_layer(filtered, style_callback)
            │
            └─▶ _update_detection_tiles()
                    └─▶ Build DataFrame from filtered features
                    └─▶ Sort by probability ascending (hardest first)
                    └─▶ tile_panel.update_results(df)
```

---

## Reset

**Trigger**: User clicks Reset button

```
reset_btn.on_event("click")
    │
    └─▶ reset_all(None)                          # app.py:1465
            │
            ├─▶ state.reset()                    # state.py:51
            │       └─▶ Clear pos_ids, neg_ids, cached_embeddings
            │       └─▶ query_vector = None
            │       └─▶ detection_mode = False
            │       └─▶ detection_data = None
            │       └─▶ detection_labels.clear()
            │
            ├─▶ map_manager.update_label_layers(empty, empty, empty)
            ├─▶ map_manager.update_search_layer(empty)
            ├─▶ map_manager.clear_detection_layer()
            ├─▶ map_manager.clear_vector_layer()
            ├─▶ map_manager.clear_highlight()
            │
            ├─▶ Hide detection_controls
            ├─▶ Reset slider to defaults (0.0 - 1.0, value 0.5)
            │
            ├─▶ tile_panel.clear()
            ├─▶ tile_panel.hide()
            │
            └─▶ _update_status()
```

---

## Related Files

- `geovibes/ui/app.py` — Event handlers and orchestration
- `geovibes/ui/state.py` — State mutations
- `geovibes/ui/map_manager.py` — Map layer updates
- `geovibes/ui/tiles.py` — Tile panel updates
- `geovibes/ui/datasets.py` — File loading
