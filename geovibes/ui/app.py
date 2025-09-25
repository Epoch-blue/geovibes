"""GeoVibes ipyleaflet application orchestrator."""

from __future__ import annotations

import json
import warnings
from typing import Dict, Optional

import ipywidgets as ipyw
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.wkt
import webbrowser
import faiss
from IPython.display import display
from ipywidgets import (
    Accordion,
    Button,
    Dropdown,
    FileUpload,
    HBox,
    HTML,
    IntSlider,
    Label,
    Layout,
    ToggleButtons,
    VBox,
)

from geovibes.ui_config import BasemapConfig, DatabaseConstants, LayerStyles, UIConstants
from geovibes.ui.data_manager import DataManager
from geovibes.ui.datasets import DatasetManager
from geovibes.ui.map_manager import MapManager
from geovibes.ui.state import AppState
from geovibes.ui.status import StatusBus
from geovibes.ui.tiles import TilePanel
from geovibes.ui.utils import log_to_file

warnings.simplefilter("ignore", category=FutureWarning)

if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn(
        "MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key.",
        RuntimeWarning,
    )


class GeoVibes:
    """Interactive map interface for geospatial similarity search."""

    @classmethod
    def create(
        cls,
        duckdb_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2025-01-01",
        gcp_project: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        return cls(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            gcp_project=gcp_project,
            verbose=verbose,
            **kwargs,
        )

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        gcp_project: Optional[str] = None,
        duckdb_connection=None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        baselayer_url: Optional[str] = None,
        enable_ee: Optional[bool] = None,
        disable_ee: bool = False,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes...")

        # Core services
        self.data = DataManager(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            gcp_project=gcp_project,
            config=config,
            config_path=config_path,
            duckdb_connection=duckdb_connection,
            baselayer_url=baselayer_url,
            enable_ee=enable_ee,
            disable_ee=disable_ee,
            verbose=verbose,
        )
        self.state = AppState()
        self.status_bus = StatusBus()
        self.map_manager = MapManager(
            data_manager=self.data,
            state=self.state,
            status_bus=self.status_bus,
            verbose=self.verbose,
        )
        self.dataset_manager = DatasetManager(
            data_manager=self.data,
            map_manager=self.map_manager,
            state=self.state,
            verbose=self.verbose,
        )
        self.tile_panel = TilePanel(
            state=self.state,
            map_manager=self.map_manager,
            on_label=self._handle_tile_label,
            on_center=self._handle_tile_center,
            verbose=self.verbose,
        )

        self._build_ui()
        self._wire_events()

        self.map_manager.update_boundary_layer(self.data.effective_boundary_path)
        self._update_layers()
        self._show_operation_status("Ready")
        self._update_status()

        display(self.main_layout)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.side_panel, self.ui_widgets = self._build_side_panel()
        self.main_layout = self.map_manager.make_layout(self.side_panel)

    def _build_side_panel(self):
        self.search_btn = Button(
            description="Search",
            layout=Layout(flex="1", height=UIConstants.BUTTON_HEIGHT),
            button_style="success",
            tooltip="Find points similar to your positive labels",
        )
        self.tiles_button = Button(
            description="",
            icon="th",
            layout=Layout(width="40px", height=UIConstants.BUTTON_HEIGHT),
            tooltip="View search results as tiles",
        )
        search_controls = HBox([self.search_btn, self.tiles_button])

        self.neighbors_slider = IntSlider(
            value=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            layout=Layout(width="100%"),
        )
        self.reset_btn = Button(
            description="🗑️ Reset",
            layout=Layout(width="100%", height=UIConstants.RESET_BUTTON_HEIGHT),
            tooltip="Clear all labels and search results",
        )
        search_section = VBox(
            [search_controls, self.neighbors_slider, self.reset_btn],
            layout=Layout(padding="5px", margin="0 0 10px 0"),
        )

        self.label_toggle = ToggleButtons(
            options=[("Positive", "Positive"), ("Negative", "Negative"), ("Erase", "Erase")],
            value="Positive",
            layout=Layout(width="100%"),
        )
        self.selection_mode = ToggleButtons(
            options=[("Point", "point"), ("Polygon", "polygon")],
            value="point",
            layout=Layout(width="100%"),
        )

        # Basemap buttons
        self.basemap_buttons = {}
        basemap_widgets = []
        for name in self.map_manager.basemap_tiles.keys():
            btn = Button(
                description=name.replace("_", " "),
                layout=Layout(width="100%", margin="1px"),
            )
            btn.basemap_name = name
            self.basemap_buttons[name] = btn
            basemap_widgets.append(btn)
        self._update_basemap_button_styles()

        # Dataset and external tools
        self.save_btn = Button(description="💾 Save Dataset", layout=Layout(width="100%"))
        self.load_btn = Button(description="📂 Load Dataset", layout=Layout(width="100%"))
        self.file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none"),
        )
        self.add_vector_btn = Button(
            description="📄 Add Vector Layer",
            layout=Layout(width="100%"),
        )
        self.vector_file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none"),
        )
        self.google_maps_btn = Button(
            description="🌍 Google Maps ↗",
            layout=Layout(width="100%"),
        )
        self.run_button = Button(
            description="Find Similar",
            button_style="primary",
            layout=Layout(width="120px"),
        )

        # Database dropdown
        database_section_widgets = []
        if getattr(self.data, "available_databases", []):
            options = [
                (
                    entry.get("display_name", entry["db_path"]),
                    entry["db_path"],
                )
                for entry in self.data.available_databases
            ]
            self.database_dropdown = Dropdown(
                options=options,
                value=self.data.current_database_path,
                layout=Layout(width="100%"),
            )
            database_section_widgets.append(Label("Select Database:"))
            database_section_widgets.append(self.database_dropdown)
        else:
            self.database_dropdown = None

        accordion_children = []
        accordion_titles = []
        if database_section_widgets:
            accordion_children.append(
                VBox(database_section_widgets, layout=Layout(padding="5px"))
            )
            accordion_titles.append("Database")

        accordion_children.extend(
            [
                VBox(
                    [
                        Label("Label Type:"),
                        self.label_toggle,
                        Label("Selection Mode:", layout=Layout(margin="10px 0 0 0")),
                        self.selection_mode,
                    ],
                    layout=Layout(padding="5px"),
                ),
                VBox(basemap_widgets, layout=Layout(padding="5px")),
                VBox(
                    [
                        self.save_btn,
                        self.load_btn,
                        self.file_upload,
                        self.add_vector_btn,
                        self.vector_file_upload,
                        self.google_maps_btn,
                        self.run_button,
                    ],
                    layout=Layout(padding="5px"),
                ),
            ]
        )
        accordion_titles.extend(["Label Mode", "Basemaps", "Export & Tools"])

        accordion = Accordion(children=accordion_children)
        for idx, title in enumerate(accordion_titles):
            accordion.set_title(idx, title)
        accordion.selected_index = 0

        self.collapse_btn = Button(
            description="◀",
            layout=Layout(width=UIConstants.COLLAPSE_BUTTON_SIZE, height=UIConstants.COLLAPSE_BUTTON_SIZE),
            tooltip="Collapse/Expand Panel",
        )
        self.panel_collapsed = False

        panel_header = HBox(
            [Label("Controls", layout=Layout(flex="1")), self.collapse_btn],
            layout=Layout(width="100%", justify_content="space-between", padding="2px"),
        )
        self.accordion_container = VBox([accordion], layout=Layout(width="100%"))

        panel = VBox(
            [panel_header, search_section, self.accordion_container],
            layout=Layout(width=UIConstants.PANEL_WIDTH, padding="5px"),
        )

        ui_widgets = {
            "search_btn": self.search_btn,
            "reset_btn": self.reset_btn,
            "label_toggle": self.label_toggle,
            "selection_mode": self.selection_mode,
            "neighbors_slider": self.neighbors_slider,
            "basemap_buttons": self.basemap_buttons,
            "save_btn": self.save_btn,
            "load_btn": self.load_btn,
            "file_upload": self.file_upload,
            "add_vector_btn": self.add_vector_btn,
            "vector_file_upload": self.vector_file_upload,
            "google_maps_btn": self.google_maps_btn,
            "collapse_btn": self.collapse_btn,
            "tiles_button": self.tiles_button,
            "run_button": self.run_button,
            "database_dropdown": self.database_dropdown,
        }
        return panel, ui_widgets

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        self.search_btn.on_click(self.search_click)
        self.run_button.on_click(self.search_click)
        self.reset_btn.on_click(self.reset_all)
        self.tiles_button.on_click(lambda _b: self.tile_panel.toggle())
        self.label_toggle.observe(self._on_label_change, names="value")
        self.selection_mode.observe(self._on_selection_mode_change, names="value")
        for name, btn in self.basemap_buttons.items():
            btn.on_click(lambda _b, basemap=name: self._on_basemap_select(basemap))
        if self.database_dropdown:
            self.database_dropdown.observe(self._on_database_change, names="value")
        self.collapse_btn.on_click(self._on_toggle_collapse)

        self.save_btn.on_click(lambda _b: self._handle_save_dataset())
        self.load_btn.on_click(
            lambda btn: DatasetManager.toggle_upload(
                btn, self.file_upload, "📂 Cancel Load", "📂 Load Dataset"
            )
        )
        self.file_upload.observe(self._on_file_upload, names="value")
        self.add_vector_btn.on_click(
            lambda btn: DatasetManager.toggle_upload(
                btn, self.vector_file_upload, "📄 Cancel Vector", "📄 Add Vector Layer"
            )
        )
        self.vector_file_upload.observe(self._on_vector_upload, names="value")
        self.google_maps_btn.on_click(self._on_google_maps_click)

        self.map_manager.register_draw_handler(self._handle_draw)
        self.map_manager.map.on_interaction(self._on_map_interaction)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_label_change(self, change) -> None:
        self.state.set_label_mode(change["new"])
        self._update_status()

    def _on_selection_mode_change(self, change) -> None:
        self.state.selection_mode = change["new"]
        self.state.lasso_mode = change["new"] == "polygon"
        self._update_status()

    def _on_basemap_select(self, basemap_name: str) -> None:
        self.map_manager.update_basemap(basemap_name)
        self.state.tile_basemap = basemap_name
        self.tile_panel.reload_tiles_for_new_basemap()
        self._update_basemap_button_styles()

    def _on_toggle_collapse(self, _button) -> None:
        if self.panel_collapsed:
            self.accordion_container.layout.display = "flex"
            self.collapse_btn.description = "◀"
            self.panel_collapsed = False
        else:
            self.accordion_container.layout.display = "none"
            self.collapse_btn.description = "▶"
            self.panel_collapsed = True

    def _on_database_change(self, change) -> None:
        new_path = change["new"]
        if new_path == self.data.current_database_path:
            return
        self._show_operation_status("🔄 Loading database...")
        try:
            self.data.switch_database(new_path)
            self.map_manager.center_on(self.data.center_y, self.data.center_x)
            self.map_manager.update_boundary_layer(self.data.effective_boundary_path)
            self.reset_all()
            if self.database_dropdown:
                self.database_dropdown.value = new_path
        except Exception as exc:
            if self.verbose:
                print(f"❌ Failed to switch database: {exc}")
            self._show_operation_status(f"❌ Failed to load database: {exc}")
        else:
            self._show_operation_status("✅ Database loaded")
        finally:
            self._update_status()

    def _on_google_maps_click(self, _button) -> None:
        lat, lon = self.map_manager.map.center
        url = f"https://www.google.com/maps/@{lat},{lon},15z"
        webbrowser.open(url, new=2)

    def _on_file_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.dataset_manager.load_from_content(content, file_info["name"])
            self._update_layers()
            self._update_query_vector()
            self._show_operation_status("✅ Dataset loaded")
        except Exception as exc:
            self._show_operation_status(f"❌ Error loading file: {exc}")
            if self.verbose:
                print(f"❌ Error loading file: {exc}")
        finally:
            self.file_upload.value = ()
            self.file_upload.layout.display = "none"
            self.load_btn.description = "📂 Load Dataset"

    def _on_vector_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.dataset_manager.add_vector_from_content(content, file_info["name"])
            self._show_operation_status("✅ Vector layer added")
        except Exception as exc:
            self._show_operation_status(f"❌ Error loading vector: {exc}")
            if self.verbose:
                print(f"❌ Error loading vector: {exc}")
        finally:
            self.vector_file_upload.value = ()
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.description = "📄 Add Vector Layer"

    def _on_map_interaction(self, **kwargs) -> None:
        lat, lon = kwargs.get("coordinates", (0, 0))
        self._update_status(lat=lat, lon=lon)

        if kwargs.get("type") != "click":
            return

        modifiers = kwargs.get("modifiers", {})
        if modifiers.get("ctrlKey"):
            webbrowser.open(f"https://www.google.com/maps/@{lat},{lon},18z", new=2)
            log_to_file("Handled as Ctrl-Click for Google Maps. Returning.")
            return

        if not self.state.execute_label_point or self.state.lasso_mode or self.state.polygon_drawing:
            return

        self.label_point(lon=lon, lat=lat)

    # ------------------------------------------------------------------
    # Labeling and drawing
    # ------------------------------------------------------------------

    def label_point(self, lon: float, lat: float) -> None:
        log_to_file("label_point: Querying database for nearest point.")
        result = self.data.nearest_point(lon, lat)
        if result is None:
            self._show_operation_status("⚠️ No points found near click.")
            return

        point_id = str(result[0])
        embedding = np.array(result[3])
        self.state.cached_embeddings[point_id] = embedding

        if self.state.select_val == UIConstants.ERASE_LABEL:
            erase_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings
            WHERE id = ?
            """
            erase_geojson = self.data.duckdb_connection.execute(
                erase_query, [point_id]
            ).fetchone()
            if erase_geojson:
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": json.loads(erase_geojson[0]),
                            "properties": {},
                        }
                    ],
                }
                self.map_manager.update_label_layers(
                    pos_geojson=self._empty_collection(),
                    neg_geojson=self._empty_collection(),
                    erase_geojson=geojson,
                )
            self.state.remove_label(point_id)
            self._show_operation_status("✅ Erased label")
        else:
            label_state = self.state.apply_label(point_id, self.state.select_val)
            status = "Positive" if label_state == "positive" else "Negative"
            if label_state == "removed":
                self._show_operation_status("✅ Removed label")
            else:
                self._show_operation_status(f"✅ Labeled point as {status}")

        self._update_layers()
        self._update_query_vector()

    def _handle_draw(self, target, action, geo_json) -> None:
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            polygon_coords = geo_json["geometry"]["coordinates"][0]
            polygon = shapely.geometry.Polygon(polygon_coords)
            point_ids: list[str] = []

            if self.state.detections_with_embeddings is not None:
                within_mask = self.state.detections_with_embeddings.geometry.within(
                    polygon
                )
                cached_points = self.state.detections_with_embeddings[within_mask]
                point_ids.extend(cached_points["id"].tolist())

            if not point_ids:
                polygon_wkt = polygon.wkt
                query = f"""
                SELECT id
                FROM geo_embeddings
                WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
                """
                arrow_table = self.data.duckdb_connection.execute(query).fetch_arrow_table()
                point_ids.extend(arrow_table.to_pandas()["id"].astype(str).tolist())

            if not point_ids:
                self._show_operation_status("⚠️ No points found within polygon")
                self.map_manager.draw_control.clear()
                self._update_status()
                return

            self._fetch_embeddings(point_ids)
            labeled = 0
            for pid in point_ids:
                result = self.state.apply_label(pid, self.state.select_val)
                if result != "removed":
                    labeled += 1
            self._show_operation_status(
                f"✅ Labeled {labeled} points as {self.state.current_label}"
            )
            self._update_layers()
            self._update_query_vector()
            self.map_manager.draw_control.clear()
        elif action == "drawstart":
            self.state.polygon_drawing = True
            self._update_status()
        elif action == "deleted":
            self.state.polygon_drawing = False
            self._update_status()

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------

    def search_click(self, _button=None) -> None:
        self.state.tile_page = 0
        if self.state.query_vector is None or len(self.state.query_vector) == 0:
            if self.verbose:
                print("🔍 No query vector. Please label some points first.")
            self._show_operation_status("⚠️ Label some points to search")
            return
        self._search_faiss()

    def _search_faiss(self) -> None:
        n_neighbors = self.neighbors_slider.value
        all_labeled = self.state.pos_ids + self.state.neg_ids
        extra_results = min(len(all_labeled), n_neighbors // 2)
        total_requested = n_neighbors + extra_results

        query_vector_np = self.state.query_vector.reshape(1, -1).astype("float32")
        params = faiss.SearchParametersIVF(nprobe=4096)
        self._show_operation_status(
            f"🔍 FAISS Search: Finding {n_neighbors} neighbors..."
        )
        distances, ids = self.data.faiss_index.search(query_vector_np, total_requested, params=params)
        faiss_ids = ids[0].tolist()
        faiss_distances = distances[0].tolist()

        if not faiss_ids:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        metadata_df = self.data.query_search_metadata(faiss_ids)
        if metadata_df is None or metadata_df.empty:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        id_map = {id_val: i for i, id_val in enumerate(faiss_ids)}
        metadata_df["sort_order"] = metadata_df["id"].map(id_map)
        metadata_df = metadata_df.sort_values("sort_order").drop(columns=["sort_order"])
        metadata_df["distance"] = faiss_distances[: len(metadata_df)]

        self._process_search_results(metadata_df, n_neighbors)

    def _process_search_results(self, results_df: pd.DataFrame, n_neighbors: int) -> None:
        all_labeled_ids = set(self.state.pos_ids + self.state.neg_ids)
        if not results_df.empty and all_labeled_ids:
            mask = ~results_df["id"].astype(str).isin(all_labeled_ids)
            filtered = results_df[mask].head(n_neighbors)
        else:
            filtered = results_df.head(n_neighbors)

        if filtered.empty:
            self._show_operation_status("✅ Search complete. No results found.")
            self.map_manager.update_search_layer(self._empty_collection())
            self.tile_panel.clear()
            return

        self._show_operation_status(f"✅ Found {len(filtered)} similar points.")

        geometries = [shapely.wkt.loads(row["geometry_wkt"]) for _, row in filtered.iterrows()]
        self.state.detections_with_embeddings = pd.DataFrame(
            {
                "id": filtered["id"].astype(str).values,
                "distance": filtered["distance"].values,
            }
        )
        self.state.detections_with_embeddings["geometry"] = geometries

        detections_geojson = {"type": "FeatureCollection", "features": []}
        min_distance = filtered["distance"].min()
        max_distance = filtered["distance"].max()
        for _, row in filtered.sort_values("distance", ascending=False).iterrows():
            color = UIConstants.distance_to_color(
                row["distance"], min_distance, max_distance
            )
            detections_geojson["features"].append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry_json"]),
                    "properties": {
                        "id": str(row["id"]),
                        "distance": row["distance"],
                        "color": color,
                        "fillColor": color,
                    },
                }
            )

        self.state.last_search_results_df = filtered.copy()
        self.map_manager.update_search_layer(
            detections_geojson,
            style_callback=self._search_style_callback,
        )
        self.tile_panel.update_results(filtered)
        self.tiles_button.button_style = "success"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_embeddings(self, point_ids):
        if not point_ids:
            return
        for chunk_df in self.data.fetch_embeddings(point_ids):
            for _, row in chunk_df.iterrows():
                point_id = str(row["id"])
                self.state.cached_embeddings[point_id] = np.array(row["embedding"])

    def _update_layers(self) -> None:
        pos_geojson = self._geojson_for_ids(self.state.pos_ids)
        neg_geojson = self._geojson_for_ids(self.state.neg_ids)
        self.map_manager.update_label_layers(
            pos_geojson=pos_geojson,
            neg_geojson=neg_geojson,
            erase_geojson=self._empty_collection(),
        )

    def _geojson_for_ids(self, ids):
        if not ids:
            return self._empty_collection()
        prepared_ids = [str(pid) for pid in ids]
        placeholders = ",".join(["?" for _ in prepared_ids])
        query = f"""
        SELECT ST_AsGeoJSON(geometry) as geometry
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        df = self.data.duckdb_connection.execute(query, prepared_ids).df()
        features = [
            {
                "type": "Feature",
                "geometry": json.loads(row["geometry"]),
                "properties": {},
            }
            for _, row in df.iterrows()
        ]
        return {"type": "FeatureCollection", "features": features}

    def _update_query_vector(self) -> None:
        if not self.state.pos_ids:
            self.state.query_vector = None
            return
        self._fetch_embeddings(self.state.pos_ids)
        if self.state.neg_ids:
            self._fetch_embeddings(self.state.neg_ids)
        self.state.update_query_vector()

    def _handle_tile_label(self, point_id: str, row, label: str) -> None:
        if point_id not in self.state.cached_embeddings:
            self._fetch_embeddings([point_id])
        result = self.state.apply_label(point_id, label)
        if result == "positive":
            self._show_operation_status("✅ Labeled tile as Positive")
        elif result == "negative":
            self._show_operation_status("✅ Labeled tile as Negative")
        else:
            self._show_operation_status("✅ Removed label from tile")
        self._update_layers()
        self._update_query_vector()

    def _handle_tile_center(self, row) -> None:
        geom = shapely.wkt.loads(row["geometry_wkt"])
        lat, lon = geom.y, geom.x
        self.map_manager.center_on(lat, lon, zoom=14)
        half_size = 0.0025 / 2
        square_coords = [
            (lon - half_size, lat - half_size),
            (lon + half_size, lat - half_size),
            (lon + half_size, lat + half_size),
            (lon - half_size, lat + half_size),
            (lon - half_size, lat - half_size),
        ]
        polygon = shapely.geometry.Polygon(square_coords)
        self.map_manager.highlight_polygon(polygon, color="red")
        self._show_operation_status("📍 Centered on tile")

    def _handle_save_dataset(self) -> None:
        result = self.dataset_manager.save_dataset()
        if result:
            self._show_operation_status("✅ Dataset saved")
        else:
            self._show_operation_status("⚠️ Nothing to save")

    def reset_all(self, _button=None) -> None:
        if self.verbose:
            print("🗑️ Resetting all labels and search results...")
        self.state.reset()
        self.map_manager.update_label_layers(
            pos_geojson=self._empty_collection(),
            neg_geojson=self._empty_collection(),
            erase_geojson=self._empty_collection(),
        )
        self.map_manager.update_search_layer(self._empty_collection())
        self.map_manager.clear_vector_layer()
        self.map_manager.clear_highlight()
        self.tile_panel.clear()
        self.tiles_button.button_style = ""
        self._clear_operation_status()
        self._update_status()

    def _search_style_callback(self, feature):
        props = feature.get("properties", {})
        return {
            "color": "black",
            "radius": UIConstants.SEARCH_POINT_RADIUS,
            "fillColor": props.get("fillColor", UIConstants.SEARCH_COLOR),
            "opacity": UIConstants.POINT_OPACITY,
            "fillOpacity": UIConstants.POINT_FILL_OPACITY,
            "weight": UIConstants.SEARCH_POINT_WEIGHT,
        }

    def _update_status(self, lat: Optional[float] = None, lon: Optional[float] = None) -> None:
        self.map_manager.update_status(lat=lat, lon=lon)

    def _show_operation_status(self, message: str) -> None:
        self.map_manager.set_operation(message)

    def _clear_operation_status(self) -> None:
        self.map_manager.clear_operation()

    def _update_basemap_button_styles(self) -> None:
        for name, btn in self.basemap_buttons.items():
            btn.button_style = "info" if name == self.map_manager.current_basemap else ""

    @staticmethod
    def _empty_collection() -> Dict:
        return {"type": "FeatureCollection", "features": []}

    def close(self) -> None:
        self.data.close()


__all__ = ["GeoVibes"]
