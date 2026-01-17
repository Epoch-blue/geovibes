"""GeoVibes ipyleaflet application orchestrator."""

from __future__ import annotations

import json
import math
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.ops
import shapely.wkt
import webbrowser
import faiss
import pyproj
from IPython.display import display
from ipywidgets import (
    Button,
    FileUpload,
    HBox,
    HTML,
    IntSlider,
    Label,
    Layout,
    VBox,
)
import ipyvuetify as v

from geovibes.ui_config import BasemapConfig, UIConstants
from geovibes.ui.data_manager import DataManager
from geovibes.ui.datasets import DatasetManager
from geovibes.ui.location_analyzer import create_location_analyzer
from geovibes.ui.map_manager import MapManager
from geovibes.ui.state import AppState
from geovibes.ui.status import StatusBus
from geovibes.ui.tiles import TilePanel
from geovibes.ui.utils import log_to_file

warnings.simplefilter("ignore", category=FutureWarning)

SIDE_PANEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.geovibes-panel,
.geovibes-panel * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Make search button more prominent */
.geovibes-panel .search-btn {
    height: 40px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.geovibes-panel .v-btn {
    text-transform: none !important;
    letter-spacing: 0.3px !important;
    font-size: 12px !important;
}

.geovibes-panel .v-btn__content {
    font-weight: 500 !important;
}

.geovibes-panel .section-label {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: block;
}

.geovibes-panel .v-card {
    margin-bottom: 8px !important;
}

.geovibes-panel .v-btn-toggle {
    width: 100%;
}

.geovibes-panel .v-btn-toggle .v-btn {
    flex: 1 !important;
    height: 32px !important;
}

.geovibes-panel .v-slider {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.geovibes-panel .v-select {
    font-size: 12px !important;
}

.geovibes-panel .v-select .v-input__slot {
    min-height: 36px !important;
}

.geovibes-panel .v-select .v-select__selection {
    font-size: 12px !important;
}

.geovibes-panel .v-list-item__title {
    font-size: 12px !important;
}

.geovibes-panel .text-body-2 {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* Compact FileUpload widget */
.geovibes-panel .widget-upload {
    padding: 0 !important;
    margin: 4px 0 !important;
}

.geovibes-panel .widget-upload > .widget-label {
    display: none !important;
}

.geovibes-panel .widget-upload-label {
    font-size: 11px !important;
    padding: 4px 8px !important;
    margin: 0 !important;
}
</style>
"""

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
        verbose: bool = False,
        enable_ee: Optional[bool] = None,
        **kwargs,
    ):
        return cls(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            enable_ee=enable_ee,
            **kwargs,
        )

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        duckdb_connection=None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        baselayer_url: Optional[str] = None,
        disable_ee: bool = False,
        verbose: bool = False,
        enable_ee: Optional[bool] = None,
        **unused_kwargs: Any,
    ) -> None:
        self.verbose = verbose
        if self.verbose:
            print("Initializing GeoVibes...")

        if "enable_ee" in unused_kwargs and self.verbose:
            print(
                "ℹ️ Pass enable_ee via config or GEOVIBES_ENABLE_EE environment variable."
            )

        # Core services
        self.data = DataManager(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            config=config,
            config_path=config_path,
            duckdb_connection=duckdb_connection,
            baselayer_url=baselayer_url,
            disable_ee=disable_ee,
            verbose=verbose,
            enable_ee=enable_ee,
        )
        self.id_column_candidates = getattr(self.data, "id_column_candidates", ["id"])
        self.external_id_column = getattr(self.data, "external_id_column", "id")
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

        # Initialize location analyzer if API keys are available
        self.location_analyzer = create_location_analyzer()

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
        css_widget = HTML(SIDE_PANEL_CSS)

        # Search section with ipyvuetify (style L: full-width search + icon button)
        self.search_btn = v.Btn(
            block=True,
            color="primary",
            depressed=True,
            class_="search-btn",
            children=[
                v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
                "Search",
            ],
        )
        self.tiles_button = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-view-grid-outline"])],
        )
        search_row = v.Row(
            no_gutters=True,
            align="center",
            class_="mb-2",
            children=[
                v.Col(cols=10, class_="pr-1", children=[self.search_btn]),
                v.Col(
                    cols=2,
                    class_="pl-1 d-flex justify-end",
                    children=[self.tiles_button],
                ),
            ],
        )

        self.neighbors_slider = v.Slider(
            v_model=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            thumb_label=True,  # Only show on drag, not always
            hide_details=True,
            class_="mt-0 flex-grow-1",
        )
        self.neighbors_label = v.Html(
            tag="span",
            class_="text-body-2 font-weight-medium ml-2",
            children=[str(UIConstants.DEFAULT_NEIGHBORS)],
            style_="min-width: 45px; text-align: right;",
        )
        slider_row = v.Row(
            no_gutters=True,
            align="center",
            children=[
                v.Col(cols=10, class_="pa-0", children=[self.neighbors_slider]),
                v.Col(
                    cols=2,
                    class_="pa-0 d-flex justify-end",
                    children=[self.neighbors_label],
                ),
            ],
        )

        search_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[search_row, slider_row],
        )

        # Label toggle using ipyvuetify BtnToggle with MDI icons
        self._label_values = ["Positive", "Negative", "Erase"]
        self.label_toggle = v.BtnToggle(
            v_model=0,
            mandatory=True,
            class_="d-flex",
            children=[
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-thumb-up-outline"])],
                ),
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-thumb-down-outline"])],
                ),
                v.Btn(
                    small=True,
                    children=[v.Icon(small=True, children=["mdi-eraser"])],
                ),
            ],
        )

        label_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["LABEL"]),
                self.label_toggle,
            ],
        )

        # Mode toggle with ipyvuetify BtnToggle
        self._mode_values = ["point", "polygon"]
        self.selection_mode = v.BtnToggle(
            v_model=0,
            mandatory=True,
            class_="d-flex",
            children=[
                v.Btn(small=True, children=["• Point"]),
                v.Btn(small=True, children=["▢ Polygon"]),
            ],
        )

        mode_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["MODE"]),
                self.selection_mode,
            ],
        )

        # Detection controls using ipyvuetify (same pattern as neighbors_slider)
        self.detection_threshold_slider = v.Slider(
            v_model=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            thumb_label=True,
            hide_details=True,
            class_="mt-0 flex-grow-1",
        )
        self.detection_threshold_label = v.Html(
            tag="span",
            class_="text-body-2 font-weight-medium ml-2",
            children=["0.50"],
            style_="min-width: 45px; text-align: right;",
        )
        detection_slider_row = v.Row(
            no_gutters=True,
            align="center",
            children=[
                v.Col(
                    cols=10, class_="pa-0", children=[self.detection_threshold_slider]
                ),
                v.Col(
                    cols=2,
                    class_="pa-0 d-flex justify-end",
                    children=[self.detection_threshold_label],
                ),
            ],
        )
        self.detection_controls = v.Card(
            outlined=True,
            class_="section-card pa-3",
            style_="display: none;",
            children=[
                v.Html(
                    tag="span",
                    class_="section-label",
                    children=["DETECTION THRESHOLD"],
                ),
                detection_slider_row,
            ],
        )

        # Database dropdown with ipyvuetify
        if getattr(self.data, "available_databases", []):
            db_items = [
                {
                    "text": entry.get("display_name", entry["db_path"]),
                    "value": entry["db_path"],
                }
                for entry in self.data.available_databases
            ]
            self.database_dropdown = v.Select(
                v_model=self.data.current_database_path,
                items=db_items,
                dense=True,
                outlined=True,
                hide_details=True,
            )
        else:
            self.database_dropdown = None

        # Basemap dropdown with ipyvuetify (same style as database dropdown)
        basemap_names = list(self.map_manager.basemap_tiles.keys())
        basemap_items = [
            {"text": name.replace("_", " "), "value": name} for name in basemap_names
        ]
        self.basemap_dropdown = v.Select(
            v_model=basemap_names[0] if basemap_names else None,
            items=basemap_items,
            dense=True,
            outlined=True,
            hide_details=True,
        )
        self.basemap_names = basemap_names

        # Export buttons with ipyvuetify (using MDI outline icons)
        self.save_btn = v.Btn(
            small=True,
            children=[
                v.Icon(small=True, children=["mdi-content-save-outline"]),
                " Save",
            ],
        )
        self.load_btn = v.Btn(
            small=True,
            children=[
                v.Icon(small=True, children=["mdi-folder-open-outline"]),
                " Load",
            ],
        )
        self.file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none", margin="4px 0 0 0"),
        )
        self.add_vector_btn = v.Btn(
            small=True,
            children=[v.Icon(small=True, children=["mdi-vector-polygon"]), " Vector"],
        )
        self.vector_file_upload = FileUpload(
            accept=".geojson,.parquet",
            multiple=False,
            layout=Layout(width="100%", display="none", margin="4px 0 0 0"),
        )
        self.google_maps_btn = v.Btn(
            small=True,
            children=[v.Icon(small=True, children=["mdi-google-maps"]), " Maps"],
        )

        # Database card (always visible)
        if self.database_dropdown:
            database_card = v.Card(
                outlined=True,
                class_="section-card pa-3",
                children=[
                    v.Html(tag="span", class_="section-label", children=["DATABASE"]),
                    self.database_dropdown,
                ],
            )
        else:
            database_card = None

        # Basemaps card (always visible)
        basemaps_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["BASEMAP"]),
                self.basemap_dropdown,
            ],
        )

        # Export & Tools card (always visible)
        # FileUpload widgets are kept outside v.Card to avoid rendering issues
        export_card = v.Card(
            outlined=True,
            class_="section-card pa-3",
            children=[
                v.Html(tag="span", class_="section-label", children=["EXPORT & TOOLS"]),
                v.BtnToggle(
                    v_model=None,
                    dense=True,
                    class_="d-flex flex-wrap",
                    children=[
                        self.save_btn,
                        self.load_btn,
                    ],
                ),
                v.BtnToggle(
                    v_model=None,
                    dense=True,
                    class_="d-flex flex-wrap mt-1",
                    children=[
                        self.add_vector_btn,
                        self.google_maps_btn,
                    ],
                ),
            ],
        )
        # Container for file uploads (placed outside v.Card for proper rendering)
        self.upload_container = VBox(
            [self.file_upload, self.vector_file_upload],
            layout=Layout(width="100%", padding="0 12px", margin="0"),
        )

        # Keep accordion_container reference for compatibility but not used
        self.accordion_container = VBox(
            [
                w
                for w in [
                    database_card,
                    basemaps_card,
                    export_card,
                    self.upload_container,
                ]
                if w is not None
            ],
            layout=Layout(width="100%"),
        )
        # Location analysis section - toggle (off by default) and commodity dropdown
        self.location_analysis_toggle = ToggleButtons(
            options=[("Off", False), ("On", True)],
            value=False,
            layout=Layout(width="100%"),
            tooltip="Enable/disable location analysis on map clicks",
        )
        self.location_analysis_commodity_dropdown = Dropdown(
            options=["coffee", "cocoa", "palm oil", "soy", "beef", "wood", "rubber"],
            value="coffee",
            description="Commodity:",
            layout=Layout(width="100%"),
            tooltip="Select commodity for EUDR compliance analysis",
        )

        # Location analysis section
        location_analysis_widgets = [
            Label("Location Analysis (Off by default)"),
            self.location_analysis_toggle,
            self.location_analysis_commodity_dropdown,
        ]

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
                VBox(location_analysis_widgets, layout=Layout(padding="5px")),
                VBox(
                    [
                        self.save_btn,
                        self.load_btn,
                        self.file_upload,
                        self.add_vector_btn,
                        self.vector_file_upload,
                        self.google_maps_btn,
                    ],
                    layout=Layout(padding="5px"),
                ),
            ]
        )
        accordion_titles.extend(
            ["Label Mode", "Basemaps", "Location Analysis", "Export & Tools"]
        )

        accordion = Accordion(children=accordion_children)
        for idx, title in enumerate(accordion_titles):
            accordion.set_title(idx, title)
        accordion.selected_index = 0

        # Reset button
        self.reset_btn = v.Btn(
            block=True,
            color="error",
            outlined=True,
            class_="mt-3 text-none",
            children=[
                v.Icon(small=True, class_="mr-1", children=["mdi-trash-can-outline"]),
                "Reset",
            ],
        )

        # Collapse button (keep ipywidgets for simplicity)
        self.collapse_btn = Button(
            description="◀",
            layout=Layout(
                width=UIConstants.COLLAPSE_BUTTON_SIZE,
                height=UIConstants.COLLAPSE_BUTTON_SIZE,
            ),
            tooltip="Collapse/Expand Panel",
        )
        self.panel_collapsed = False

        # Wrap in VBox with ipyvuetify components
        panel = VBox(
            [
                css_widget,
                search_card,
                label_card,
                mode_card,
                self.detection_controls,
                self.accordion_container,
                self.reset_btn,
            ],
            layout=Layout(
                width=UIConstants.PANEL_WIDTH, padding="8px", overflow="hidden"
            ),
        )
        panel.add_class("geovibes-panel")

        ui_widgets = {
            "search_btn": self.search_btn,
            "reset_btn": self.reset_btn,
            "label_toggle": self.label_toggle,
            "selection_mode": self.selection_mode,
            "neighbors_slider": self.neighbors_slider,
            "basemap_dropdown": self.basemap_dropdown,
            "save_btn": self.save_btn,
            "load_btn": self.load_btn,
            "file_upload": self.file_upload,
            "add_vector_btn": self.add_vector_btn,
            "vector_file_upload": self.vector_file_upload,
            "detection_threshold_slider": self.detection_threshold_slider,
            "google_maps_btn": self.google_maps_btn,
            "location_analysis_toggle": self.location_analysis_toggle,
            "location_analysis_commodity_dropdown": self.location_analysis_commodity_dropdown,
            "collapse_btn": self.collapse_btn,
            "tiles_button": self.tiles_button,
            "database_dropdown": self.database_dropdown,
        }
        return panel, ui_widgets

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        # ipyvuetify buttons use on_event instead of on_click
        self.search_btn.on_event("click", lambda *args: self.search_click(None))
        self.reset_btn.on_event("click", lambda *args: self.reset_all(None))
        self.tiles_button.on_event("click", lambda *args: self.tile_panel.toggle())

        # Label toggle uses v_model (index)
        self.label_toggle.observe(self._on_label_toggle_change, names="v_model")

        # BtnToggle uses v_model (index) instead of value
        self.selection_mode.observe(self._on_selection_mode_change, names="v_model")

        # Slider label update
        self.neighbors_slider.observe(self._on_neighbors_slider_change, names="v_model")

        # Basemap dropdown uses v_model (value)
        self.basemap_dropdown.observe(self._on_basemap_dropdown_change, names="v_model")

        # Database dropdown uses v_model
        if self.database_dropdown:
            self.database_dropdown.observe(self._on_database_change, names="v_model")

        self.collapse_btn.on_click(self._on_toggle_collapse)

        # Export buttons
        self.save_btn.on_event("click", lambda *args: self._handle_save_dataset())
        self.load_btn.on_event(
            "click",
            lambda *args: self._toggle_vuetify_upload(
                self.load_btn,
                self.file_upload,
                [v.Icon(small=True, children=["mdi-close"]), " Cancel"],
                [v.Icon(small=True, children=["mdi-folder-open-outline"]), " Load"],
            ),
        )
        self.file_upload.observe(self._on_file_upload, names="value")
        self.add_vector_btn.on_event(
            "click",
            lambda *args: self._toggle_vuetify_upload(
                self.add_vector_btn,
                self.vector_file_upload,
                [v.Icon(small=True, children=["mdi-close"]), " Cancel"],
                [v.Icon(small=True, children=["mdi-vector-polygon"]), " Vector"],
            ),
        )
        self.vector_file_upload.observe(self._on_vector_upload, names="value")
        self.detection_threshold_slider.observe(
            self._on_detection_threshold_change, names="v_model"
        )
        self.google_maps_btn.on_event(
            "click", lambda *args: self._on_google_maps_click(None)
        )
        self.location_analysis_toggle.observe(
            self._on_location_analysis_toggle_change, names="value"
        )
        self.location_analysis_commodity_dropdown.observe(
            self._on_commodity_change, names="value"
        )

        self.map_manager.register_draw_handler(self._handle_draw)
        self.map_manager.map.on_interaction(self._on_map_interaction)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_label_toggle_change(self, change) -> None:
        idx = change["new"]
        if idx is not None and 0 <= idx < len(self._label_values):
            value = self._label_values[idx]
            self.state.set_label_mode(value)
            self._update_status()

    def _on_selection_mode_change(self, change) -> None:
        # v_model gives us an index, convert to value
        idx = change["new"]
        if idx is not None and 0 <= idx < len(self._mode_values):
            value = self._mode_values[idx]
            self.state.selection_mode = value
            self.state.lasso_mode = value == "polygon"
            self.state.execute_label_point = value != "polygon"
            self._update_status()

    def _toggle_vuetify_upload(
        self, btn, file_upload, cancel_children, default_children
    ) -> None:
        if file_upload.layout.display == "none":
            file_upload.layout.display = "block"
            btn.children = cancel_children
        else:
            file_upload.layout.display = "none"
            btn.children = default_children

    def _on_neighbors_slider_change(self, change) -> None:
        value = change["new"]
        if value is not None:
            self.neighbors_label.children = [f"{value:,}"]

    def _on_basemap_dropdown_change(self, change) -> None:
        basemap_name = change["new"]
        if basemap_name:
            self.map_manager.update_basemap(basemap_name)
            self.tile_panel.handle_map_basemap_change(basemap_name)

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
            self.id_column_candidates = getattr(
                self.data, "id_column_candidates", ["id"]
            )
            self.external_id_column = getattr(self.data, "external_id_column", "id")
            self.map_manager.center_on(self.data.center_y, self.data.center_x)
            self.map_manager.update_boundary_layer(self.data.effective_boundary_path)
            self.reset_all()
            if self.database_dropdown:
                self.database_dropdown.v_model = new_path
        except Exception as exc:
            if self.verbose:
                print(f"❌ Failed to switch database: {exc}")
            self._show_operation_status(f"❌ Failed to load database: {exc}")
        else:
            self._show_operation_status("✅ Database loaded")
        finally:
            self._update_status()

    def _on_detection_threshold_change(self, change) -> None:
        if not self.state.detection_mode or not self.state.detection_data:
            return
        threshold = change["new"]
        self.detection_threshold_label.children = [f"{threshold:.2f}"]
        self._filter_detection_layer(threshold)
        self._update_detection_tiles()

    def _on_google_maps_click(self, _button) -> None:
        lat, lon = self.map_manager.map.center
        url = f"https://www.google.com/maps/@{lat},{lon},15z"
        webbrowser.open(url, new=2)

    def _on_location_analysis_toggle_change(self, change) -> None:
        """Handle location analysis toggle change."""
        self.state.location_analysis_enabled = change["new"]
        if self.verbose:
            status = "enabled" if change["new"] else "disabled"
            print(f"Location analysis {status}")

    def _on_commodity_change(self, change) -> None:
        """Handle commodity dropdown change."""
        self.state.location_analysis_commodity = change["new"]
        if self.verbose:
            print(f"Location analysis commodity: {change['new']}")

    def _on_file_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.reset_all()
            self.dataset_manager.load_from_content(content, file_info["name"])
            if self.state.detection_mode:
                # Show detection controls
                self.detection_controls.style_ = ""
                features = self.state.detection_data.get("features", [])
                num_detections = len(features)

                # Set slider min/max based on dataset probability range
                if features:
                    probs = [
                        f.get("properties", {}).get("probability", 0.5)
                        for f in features
                    ]
                    min_prob = min(probs)
                    max_prob = max(probs)
                    self._detection_prob_min = min_prob
                    self._detection_prob_max = max_prob
                    self.detection_threshold_slider.min = min_prob
                    self.detection_threshold_slider.max = max_prob
                    self.detection_threshold_slider.v_model = min_prob
                    self.detection_threshold_label.children = [f"{min_prob:.2f}"]

                self._show_operation_status(
                    f"🔍 Detection mode: {num_detections} detections loaded. "
                    "Click to label as negative/positive."
                )
                # Apply initial filtering and populate tile panel
                self._filter_detection_layer(self.detection_threshold_slider.v_model)
                self._update_detection_tiles()
            else:
                self.detection_controls.style_ = "display: none;"
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
            self.load_btn.children = [
                v.Icon(small=True, children=["mdi-folder-open-outline"]),
                " Load",
            ]

    def _on_vector_upload(self, change) -> None:
        if not change["new"]:
            return
        file_info = change["new"][0]
        content = DatasetManager.read_upload_content(file_info["content"])
        try:
            self.dataset_manager.add_vector_from_content(content, file_info["name"])

            # Auto-label points that intersect with uploaded geometries
            self._auto_label_from_vector_layer(content, file_info["name"])

            self._show_operation_status("✅ Vector layer added")
        except Exception as exc:
            self._show_operation_status(f"❌ Error loading vector: {exc}")
            if self.verbose:
                print(f"❌ Error loading vector: {exc}")
        finally:
            self.vector_file_upload.value = ()
            self.vector_file_upload.layout.display = "none"
            self.add_vector_btn.children = [
                v.Icon(small=True, children=["mdi-vector-polygon"]),
                " Vector",
            ]

    def _auto_label_from_vector_layer(self, content: bytes, filename: str) -> None:
        """Automatically label points that intersect with uploaded vector layer."""
        print(f"[AUTO-LABEL] Starting auto-labeling from vector layer: {filename}")
        import sys

        sys.stdout.flush()

        try:
            import geopandas as gpd
            import io

            if filename.lower().endswith(".geojson"):
                geojson_data = json.loads(content.decode("utf-8"))
                gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
                print(f"[AUTO-LABEL] Loaded GeoJSON with {len(gdf)} features")
            elif filename.lower().endswith(".parquet"):
                gdf = gpd.read_parquet(io.BytesIO(content))
                print(f"[AUTO-LABEL] Loaded Parquet with {len(gdf)} features")
            else:
                error_msg = f"[AUTO-LABEL] Unsupported file format: {filename}"
                print(error_msg)
                sys.stdout.flush()
                self._show_operation_status(error_msg)
                return

            sys.stdout.flush()

            if gdf.empty:
                error_msg = "[AUTO-LABEL] Uploaded vector layer is empty"
                print(error_msg)
                sys.stdout.flush()
                self._show_operation_status(error_msg)
                return

            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
                print("[AUTO-LABEL] Set CRS to EPSG:4326")
            elif gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                print("[AUTO-LABEL] Reprojected to EPSG:4326")
            sys.stdout.flush()

            boundary_geom = None
            if self.data.effective_boundary_path:
                try:
                    print(
                        f"[AUTO-LABEL] Boundary path: {self.data.effective_boundary_path}"
                    )
                    sys.stdout.flush()
                    boundary_gdf = gpd.read_file(self.data.effective_boundary_path)
                    if boundary_gdf.crs is None:
                        boundary_gdf.set_crs("EPSG:4326", inplace=True)
                    elif boundary_gdf.crs != "EPSG:4326":
                        boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
                    boundary_geom = boundary_gdf.unary_union
                    print(
                        f"[AUTO-LABEL] Using database boundary to filter polygons (boundary has {len(boundary_gdf)} feature(s))"
                    )
                    sys.stdout.flush()
                except Exception as e:
                    error_msg = (
                        f"[AUTO-LABEL] Could not load boundary for filtering: {e}"
                    )
                    print(error_msg)
                    import traceback

                    traceback.print_exc()
                    sys.stdout.flush()
                    boundary_geom = None
            else:
                print(
                    "[AUTO-LABEL] No database boundary available - processing all polygons"
                )
                sys.stdout.flush()

            if boundary_geom is not None:
                original_count = len(gdf)
                gdf = gdf[gdf.geometry.intersects(boundary_geom)]
                print(
                    f"[AUTO-LABEL] Filtered from {original_count} to {len(gdf)} features within boundary"
                )
                sys.stdout.flush()
                if gdf.empty:
                    error_msg = "[AUTO-LABEL] No features in uploaded layer intersect with database boundary"
                    print(error_msg)
                    sys.stdout.flush()
                    self._show_operation_status(
                        "ℹ️ No features intersect with database boundary - no points labeled"
                    )
                    return

            has_polygons = gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any()
            has_points = gdf.geom_type.isin(["Point", "MultiPoint"]).any()

            print(
                f"[AUTO-LABEL] Vector layer analysis: {len(gdf)} features, has_polygons={has_polygons}, has_points={has_points}"
            )
            sys.stdout.flush()

            labeled_count = 0

            if has_polygons:
                polygons_gdf = gdf[
                    gdf.geom_type.isin(["Polygon", "MultiPolygon"])
                ].copy()

                print(
                    f"[AUTO-LABEL] Processing {len(polygons_gdf)} polygon(s) (within boundary)"
                )
                sys.stdout.flush()

                for idx in polygons_gdf.index:
                    geom = polygons_gdf.loc[idx, "geometry"]
                    if not geom.is_valid or geom.is_empty:
                        if geom.is_empty:
                            print(
                                f"[AUTO-LABEL] Empty geometry at index {idx}, skipping..."
                            )
                            sys.stdout.flush()
                            polygons_gdf.drop(idx, inplace=True)
                            continue
                        print(
                            f"[AUTO-LABEL] Invalid geometry at index {idx}, attempting to fix..."
                        )
                        sys.stdout.flush()
                        try:
                            fixed_geom = geom.buffer(0)
                            if fixed_geom.is_valid and not fixed_geom.is_empty:
                                polygons_gdf.loc[idx, "geometry"] = fixed_geom
                                print(f"[AUTO-LABEL] Fixed geometry at index {idx}")
                                sys.stdout.flush()
                            else:
                                print(
                                    f"[AUTO-LABEL] Could not fix geometry at index {idx} (still invalid or empty), skipping..."
                                )
                                sys.stdout.flush()
                                polygons_gdf.drop(idx, inplace=True)
                        except Exception as e:
                            print(
                                f"[AUTO-LABEL] Error fixing geometry at index {idx}: {e}, skipping..."
                            )
                            sys.stdout.flush()
                            polygons_gdf.drop(idx, inplace=True)

                if len(polygons_gdf) == 0:
                    error_msg = "[AUTO-LABEL] No valid polygons after fixing geometries"
                    print(error_msg)
                    sys.stdout.flush()
                    self._show_operation_status(error_msg)
                    return

                t0 = time.time()
                if len(polygons_gdf) == 1:
                    geom = polygons_gdf.iloc[0].geometry
                    if not geom.is_valid:
                        geom = geom.buffer(0)

                    if geom.is_empty or not geom.is_valid:
                        error_msg = (
                            "[AUTO-LABEL] Polygon is empty or invalid after fixing"
                        )
                        print(error_msg)
                        sys.stdout.flush()
                        self._show_operation_status(error_msg)
                        return

                    try:
                        polygon_wkt = geom.wkt
                        if (
                            polygon_wkt is None
                            or not isinstance(polygon_wkt, str)
                            or not polygon_wkt.strip()
                        ):
                            error_msg = "[AUTO-LABEL] Invalid WKT for polygon"
                            print(error_msg)
                            sys.stdout.flush()
                            self._show_operation_status(error_msg)
                            return
                        if not polygon_wkt.upper().startswith("POLYGON"):
                            error_msg = f"[AUTO-LABEL] Invalid WKT for polygon (doesn't start with POLYGON): {polygon_wkt[:50]}"
                            print(error_msg)
                            sys.stdout.flush()
                            self._show_operation_status(error_msg)
                            return
                    except Exception as e:
                        error_msg = f"[AUTO-LABEL] Error converting polygon to WKT: {e}"
                        print(error_msg)
                        import traceback

                        traceback.print_exc()
                        sys.stdout.flush()
                        self._show_operation_status(error_msg)
                        return

                    polygon_wkt_escaped = polygon_wkt.replace("'", "''")
                    query = f"""
                    SELECT id, ST_AsText(geometry) as wkt, embedding
                    FROM geo_embeddings
                    WHERE ST_Intersects(geometry, ST_GeomFromText('{polygon_wkt_escaped}'))
                    """
                    try:
                        results = self.data.duckdb_connection.execute(query).fetchall()
                        t1 = time.time()
                        query_ms = (t1 - t0) * 1000
                        print(
                            f"[AUTO-LABEL] Found {len(results)} points intersecting polygon ({query_ms:.1f}ms)"
                        )
                        sys.stdout.flush()
                    except Exception as e:
                        error_msg = f"[AUTO-LABEL] Error querying polygon: {e}"
                        print(error_msg)
                        import traceback

                        traceback.print_exc()
                        sys.stdout.flush()
                        results = []
                else:
                    print(
                        f"[AUTO-LABEL] Querying points intersecting {len(polygons_gdf)} polygons (querying separately)..."
                    )
                    sys.stdout.flush()

                    all_results = []
                    seen_point_ids = set()

                    for idx, row in polygons_gdf.iterrows():
                        geom = row.geometry
                        if not geom.is_valid:
                            try:
                                geom = geom.buffer(0)
                            except Exception as e:
                                print(
                                    f"[AUTO-LABEL] Error fixing geometry at index {idx}: {e}, skipping..."
                                )
                                sys.stdout.flush()
                                continue

                        if geom.is_empty or not geom.is_valid:
                            print(
                                f"[AUTO-LABEL] Skipping empty/invalid geometry at index {idx}"
                            )
                            sys.stdout.flush()
                            continue

                        try:
                            polygon_wkt = geom.wkt
                            if (
                                polygon_wkt is None
                                or not isinstance(polygon_wkt, str)
                                or not polygon_wkt.strip()
                            ):
                                print(
                                    f"[AUTO-LABEL] Skipping polygon at index {idx} - invalid WKT"
                                )
                                sys.stdout.flush()
                                continue
                            if not polygon_wkt.upper().startswith("POLYGON"):
                                print(
                                    f"[AUTO-LABEL] Skipping polygon at index {idx} - WKT doesn't start with POLYGON: {polygon_wkt[:50]}"
                                )
                                sys.stdout.flush()
                                continue
                        except Exception as e:
                            print(
                                f"[AUTO-LABEL] Error converting polygon at index {idx} to WKT: {e}, skipping..."
                            )
                            import traceback

                            traceback.print_exc()
                            sys.stdout.flush()
                            continue

                        polygon_wkt_escaped = polygon_wkt.replace("'", "''")
                        query = f"""
                        SELECT id, ST_AsText(geometry) as wkt, embedding
                        FROM geo_embeddings
                        WHERE ST_Intersects(geometry, ST_GeomFromText('{polygon_wkt_escaped}'))
                        """
                        try:
                            polygon_results = self.data.duckdb_connection.execute(
                                query
                            ).fetchall()
                            if len(polygon_results) > 0:
                                print(
                                    f"[AUTO-LABEL] Polygon {idx}: Found {len(polygon_results)} points"
                                )
                                sys.stdout.flush()
                            for point_id, wkt_geom, embedding in polygon_results:
                                point_id = str(point_id)
                                if point_id not in seen_point_ids:
                                    all_results.append((point_id, wkt_geom, embedding))
                                    seen_point_ids.add(point_id)
                        except Exception as e:
                            error_msg = f"[AUTO-LABEL] Error querying polygon at index {idx}: {e}"
                            print(error_msg)
                            import traceback

                            traceback.print_exc()
                            sys.stdout.flush()
                            continue

                    t1 = time.time()
                    query_ms = (t1 - t0) * 1000
                    results = all_results
                    print(
                        f"[AUTO-LABEL] Found {len(results)} unique points intersecting {len(polygons_gdf)} polygons ({query_ms:.1f}ms)"
                    )
                    sys.stdout.flush()

                t2 = time.time()
                for point_id, wkt_geom, embedding in results:
                    point_id = str(point_id)
                    if point_id not in self.state.pos_ids:
                        self.state.cached_embeddings[point_id] = np.array(embedding)
                        geom = shapely.wkt.loads(wkt_geom)
                        self.state.cached_geometries[point_id] = (
                            shapely.geometry.mapping(geom)
                        )
                        self.state.pos_ids.append(point_id)
                        labeled_count += 1
                t3 = time.time()
                cache_ms = (t3 - t2) * 1000
                if labeled_count > 0:
                    print(
                        f"[AUTO-LABEL] Cached {labeled_count} embeddings and geometries ({cache_ms:.1f}ms)"
                    )
                    sys.stdout.flush()

            if has_points:
                points_gdf = gdf[gdf.geom_type.isin(["Point", "MultiPoint"])]
                if self.verbose:
                    print(
                        f"📊 Processing {len(points_gdf)} point(s) - finding nearby points within 100m..."
                    )

                for idx, row in points_gdf.iterrows():
                    point_wkt = row.geometry.wkt

                    query = """
                    SELECT id, ST_AsText(geometry) as wkt, embedding,
                           ST_Distance(geometry, ST_GeomFromText(?)) as dist_m
                    FROM geo_embeddings
                    WHERE ST_Distance(geometry, ST_GeomFromText(?)) <= 100
                    """
                    results = self.data.duckdb_connection.execute(
                        query, [point_wkt, point_wkt]
                    ).fetchall()
                    if self.verbose:
                        print(
                            f"  📊 Found {len(results)} points within 100m of point {idx}"
                        )

                    for point_id, wkt_geom, embedding, dist_m in results:
                        point_id = str(point_id)
                        if point_id not in self.state.pos_ids:
                            self.state.cached_embeddings[point_id] = np.array(embedding)
                            geom = shapely.wkt.loads(wkt_geom)
                            self.state.cached_geometries[point_id] = (
                                shapely.geometry.mapping(geom)
                            )
                            self.state.pos_ids.append(point_id)
                            labeled_count += 1

            if labeled_count > 0:
                t4 = time.time()
                self._update_layers()
                t5 = time.time()
                self._update_query_vector()
                t6 = time.time()
                layers_ms = (t5 - t4) * 1000
                vector_ms = (t6 - t5) * 1000
                total_ms = (t6 - t0) * 1000
                print(
                    f"[AUTO-LABEL] Updated layers ({layers_ms:.1f}ms) and query vector ({vector_ms:.1f}ms)"
                )
                print(
                    f"[AUTO-LABEL] TOTAL: Auto-labeled {labeled_count} points from vector layer ({total_ms:.1f}ms)"
                )
                sys.stdout.flush()

                success_msg = (
                    f"✅ Auto-labeled {labeled_count} points from vector layer"
                )
                self._show_operation_status(success_msg)
            else:
                error_msg = "[AUTO-LABEL] No points found to label from vector layer"
                print(error_msg)
                if has_polygons:
                    print(
                        "[AUTO-LABEL] Tip: Make sure your polygons intersect with points in the database"
                    )
                    print(
                        "[AUTO-LABEL] Tip: Check that the polygons cover areas where points exist"
                    )
                if has_points:
                    print(
                        "[AUTO-LABEL] Tip: Make sure your points are within 100m of database points"
                    )
                sys.stdout.flush()
                self._show_operation_status(
                    "ℹ️ No points found to auto-label from vector layer"
                )

        except Exception as e:
            error_msg = f"[AUTO-LABEL] Error auto-labeling from vector layer: {e}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            import sys

            sys.stdout.flush()
            self._show_operation_status(error_msg)

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

        if self.state.detection_mode:
            self._handle_detection_click(lon, lat)
            return

        if self.state.location_analysis_enabled and self.location_analyzer:
            self.analyze_location(lat, lon, self.state.location_analysis_commodity)
            return

        if (
            not self.state.execute_label_point
            or self.state.lasso_mode
            or self.state.polygon_drawing
        ):
            return

        self.label_point(lon=lon, lat=lat)

    # ------------------------------------------------------------------
    # Labeling and drawing
    # ------------------------------------------------------------------

    def label_point(self, lon: float, lat: float) -> None:
        t0 = time.time()
        log_to_file("label_point: Querying database for nearest point.")
        result = self.data.nearest_point(lon, lat)
        t1 = time.time()
        nearest_ms = (t1 - t0) * 1000
        msg = f"[TIMING] nearest_point: {nearest_ms:.1f}ms"
        print(msg)
        import sys

        sys.stdout.flush()
        self._show_operation_status(msg)  # Show in UI too

        if result is None:
            self._show_operation_status("⚠️ No points found near click.")
            return

        point_id = str(result[0])
        wkt_geometry = result[1]  # Already have geometry - cache it
        embedding = np.array(result[3])
        self.state.cached_embeddings[point_id] = embedding

        # Cache geometry to avoid querying it later
        t2 = time.time()
        # Handle case where wkt_geometry might be None or empty
        if wkt_geometry and isinstance(wkt_geometry, str) and wkt_geometry.strip():
            try:
                geom = shapely.wkt.loads(wkt_geometry)
                self.state.cached_geometries[point_id] = shapely.geometry.mapping(geom)
            except Exception:
                # If WKT parsing fails, we'll need to query geometry later
                if self.verbose:
                    print(f"⚠️ Could not parse WKT geometry for point {point_id}")
        else:
            # No WKT available - will need to query geometry if needed
            if self.verbose:
                print(f"⚠️ No WKT geometry available for point {point_id}")
        t3 = time.time()
        geom_ms = (t3 - t2) * 1000
        print(f"[TIMING] geometry cache: {geom_ms:.1f}ms")
        sys.stdout.flush()

        status = None
        if self.state.select_val == UIConstants.ERASE_LABEL:
            # Use cached geometry instead of querying
            if point_id in self.state.cached_geometries:
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": self.state.cached_geometries[point_id],
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
            status = "Erased"
        else:
            label_state = self.state.apply_label(point_id, self.state.select_val)
            status = "Positive" if label_state == "positive" else "Negative"
            if label_state == "removed":
                status = "Removed"

        t4 = time.time()
        self._update_layers()
        t5 = time.time()
        layers_ms = (t5 - t4) * 1000
        print(f"[TIMING] _update_layers: {layers_ms:.1f}ms")
        sys.stdout.flush()

        t6 = time.time()
        self._update_query_vector()
        t7 = time.time()
        vector_ms = (t7 - t6) * 1000
        total_ms = (t7 - t0) * 1000
        print(f"[TIMING] _update_query_vector: {vector_ms:.1f}ms")
        print(f"[TIMING] TOTAL label_point: {total_ms:.1f}ms")
        sys.stdout.flush()

        # ALWAYS show timing in status message
        timing_msg = f"⏱️ {total_ms:.0f}ms total (nearest: {nearest_ms:.0f}ms, layers: {layers_ms:.0f}ms, vector: {vector_ms:.0f}ms)"
        if status == "Positive":
            self._show_operation_status(f"✅ Labeled point as {status} | {timing_msg}")
        elif status == "Negative":
            self._show_operation_status(f"✅ Labeled point as {status} | {timing_msg}")
        elif status == "Erased":
            self._show_operation_status(f"✅ Erased label | {timing_msg}")
        elif status == "Removed":
            self._show_operation_status(f"✅ Removed label | {timing_msg}")
        else:
            self._show_operation_status(timing_msg)

    def analyze_location(
        self, lat: float, lon: float, user_search_context: str = ""
    ) -> Optional[Dict]:
        """
        Analyze a location using Google Places API and Gemini AI.

        Args:
            lat: Latitude
            lon: Longitude
            user_search_context: Optional search context (commodity) for better analysis

        Returns:
            Dictionary with place info, nearby places, and AI analysis, or None if not available
        """
        if not self.location_analyzer:
            with self.map_manager.location_analysis_output:
                print("⚠️ Location analyzer not available (missing API keys)")
            return None

        try:
            # Clear previous output
            self.map_manager.location_analysis_output.clear_output(wait=True)

            # Show loading message
            with self.map_manager.location_analysis_output:
                print(
                    f"🔍 Analyzing location ({lat:.4f}, {lon:.4f}) for {user_search_context}..."
                )
                import sys

                sys.stdout.flush()

            result = self.location_analyzer.analyze_location(
                lat, lon, user_search_context
            )

            if result.get("success"):
                analysis = result.get("gemini_analysis", "")
                place_info = result.get("place_info", {})
                nearby_places = result.get("nearby_places", [])

                # Display full analysis in the output widget below the map
                with self.map_manager.location_analysis_output:
                    print("\n" + "=" * 80)
                    print(f"📍 Location Analysis for {user_search_context.upper()}")
                    print(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")
                    print("=" * 80)
                    print(analysis)
                    print("\n" + "=" * 80 + "\n")
                    import sys

                    sys.stdout.flush()

                # Update status bar with brief info
                status_msg = (
                    f"✅ Location analyzed: {place_info.get('place_name', 'Location')}"
                )
                self._show_operation_status(status_msg)

                return result
            else:
                error = result.get("error", "Unknown error")
                with self.map_manager.location_analysis_output:
                    print(f"⚠️ Location analysis returned no results: {error}")
                    import sys

                    sys.stdout.flush()
                self._show_operation_status(f"⚠️ Location analysis returned no results")
                return None

        except Exception as e:
            error_msg = f"⚠️ Error analyzing location: {e}"
            with self.map_manager.location_analysis_output:
                print(error_msg)
                import sys

                sys.stdout.flush()
            if self.verbose:
                self._show_operation_status(error_msg)
            return None

    def _handle_detection_click(self, lon: float, lat: float) -> None:
        if not self.state.detection_data:
            return

        click_point = shapely.geometry.Point(lon, lat)
        features = self.state.detection_data.get("features", [])

        for feature in features:
            geom = shapely.geometry.shape(feature["geometry"])
            if geom.contains(click_point):
                props = feature.get("properties", {})
                tile_id = props.get("tile_id", props.get("id", "unknown"))
                probability = props.get("probability", 0.0)

                current_label = self.state.detection_labels.get(tile_id)
                if self.state.select_val == UIConstants.POSITIVE_LABEL:
                    new_label = 1
                    label_name = "positive (confirmed)"
                elif self.state.select_val == UIConstants.NEGATIVE_LABEL:
                    new_label = 0
                    label_name = "negative (hard negative)"
                else:
                    if tile_id in self.state.detection_labels:
                        del self.state.detection_labels[tile_id]
                        self._show_operation_status(
                            f"✅ Removed label from detection (P={probability:.2f})"
                        )
                    return

                if current_label == new_label:
                    del self.state.detection_labels[tile_id]
                    self._show_operation_status(
                        f"✅ Toggled off {label_name} (P={probability:.2f})"
                    )
                else:
                    self.state.label_detection(tile_id, new_label)
                    num_labeled = len(self.state.detection_labels)
                    self._show_operation_status(
                        f"✅ Marked as {label_name} (P={probability:.2f}) | "
                        f"Total labeled: {num_labeled}"
                    )
                self._refresh_detection_layer()
                return

        self._show_operation_status("⚠️ No detection at click location")

    def _handle_draw(self, target, action, geo_json) -> None:
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            polygon_coords = geo_json["geometry"]["coordinates"][0]
            polygon = shapely.geometry.Polygon(polygon_coords)

            # Detection mode: label detections within polygon
            if self.state.detection_mode and self.state.detection_data:
                self._label_detections_in_polygon(polygon)
                self.map_manager.draw_control.clear()
                return

            # Normal mode: label points within polygon
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
                arrow_table = self.data.duckdb_connection.execute(
                    query
                ).fetch_arrow_table()
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

    def _label_detections_in_polygon(self, polygon: shapely.geometry.Polygon) -> None:
        """Label all detections within the given polygon."""
        features = self.state.detection_data.get("features", [])
        labeled_count = 0

        # Determine label based on current mode
        if self.state.select_val == UIConstants.POSITIVE_LABEL:
            new_label = 1
            label_name = "positive"
        elif self.state.select_val == UIConstants.NEGATIVE_LABEL:
            new_label = 0
            label_name = "negative"
        else:
            # Erase mode: remove labels from detections in polygon
            for feature in features:
                geom = shapely.geometry.shape(feature["geometry"])
                if polygon.contains(geom.centroid) or polygon.intersects(geom):
                    props = feature.get("properties", {})
                    tile_id = props.get("tile_id", props.get("id", "unknown"))
                    if tile_id in self.state.detection_labels:
                        del self.state.detection_labels[tile_id]
                        labeled_count += 1
            self._show_operation_status(
                f"✅ Removed labels from {labeled_count} detections"
            )
            self._refresh_detection_layer()
            return

        for feature in features:
            geom = shapely.geometry.shape(feature["geometry"])
            # Check if detection centroid is within polygon or polygon intersects detection
            if polygon.contains(geom.centroid) or polygon.intersects(geom):
                props = feature.get("properties", {})
                tile_id = props.get("tile_id", props.get("id", "unknown"))
                self.state.label_detection(tile_id, new_label)
                labeled_count += 1

        total_labeled = len(self.state.detection_labels)
        self._show_operation_status(
            f"✅ Labeled {labeled_count} detections as {label_name} | Total: {total_labeled}"
        )
        self._refresh_detection_layer()

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------

    def search_click(self, _button=None) -> None:
        self.state.tile_page = 0
        self._reset_tiles_button()
        if self.state.query_vector is None or len(self.state.query_vector) == 0:
            if self.verbose:
                print("🔍 No query vector. Please label some points first.")
            self._show_operation_status("⚠️ Label some points to search")
            return
        self._search_faiss()

    def _search_faiss(self) -> None:
        n_neighbors = self.neighbors_slider.v_model
        all_labeled = self.state.pos_ids + self.state.neg_ids
        extra_results = min(len(all_labeled), n_neighbors // 2)
        total_requested = n_neighbors + extra_results

        query_vector_np = self.state.query_vector.reshape(1, -1).astype("float32")
        params = faiss.SearchParametersIVF(nprobe=4096)
        self._show_operation_status(
            f"🔍 FAISS Search: Finding {n_neighbors} neighbors..."
        )
        distances, ids = self.data.faiss_index.search(
            query_vector_np, total_requested, params=params
        )
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

    def _process_search_results(
        self, results_df: pd.DataFrame, n_neighbors: int
    ) -> None:
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

        geometries = [
            shapely.wkt.loads(row["geometry_wkt"]) for _, row in filtered.iterrows()
        ]
        display_column = getattr(self, "external_id_column", "id")
        base_columns = ["id", "distance"]
        if display_column != "id" and display_column in filtered.columns:
            base_columns.append(display_column)
        detections_df = filtered[base_columns].copy()
        detections_df["id"] = detections_df["id"].astype(str)
        if display_column in detections_df.columns:
            detections_df[display_column] = detections_df[display_column].astype(str)
        self.state.detections_with_embeddings = gpd.GeoDataFrame(
            detections_df,
            geometry=geometries,
            crs="EPSG:4326",
        )

        detections_geojson = {"type": "FeatureCollection", "features": []}
        min_distance = filtered["distance"].min()
        max_distance = filtered["distance"].max()
        highlight_cutoff = None
        if len(filtered) > 0:
            top_count = max(1, min(100, int(math.ceil(len(filtered) * 0.1))))
            highlight_cutoff = filtered.nsmallest(top_count, "distance")[
                "distance"
            ].max()
        for _, row in filtered.sort_values("distance", ascending=False).iterrows():
            color = UIConstants.distance_to_color(
                row["distance"], min_distance, max_distance, highlight_cutoff
            )
            display_id = self._display_id_from_row(row)
            props = {
                "id": str(row["id"]),
                "distance": row["distance"],
                "color": color,
                "fillColor": color,
                "source_id": display_id,
            }
            external_column_name = getattr(self, "external_id_column", "id")
            if external_column_name != "id" and external_column_name in row.index:
                props[external_column_name] = display_id
            detections_geojson["features"].append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry_json"]),
                    "properties": props,
                }
            )

        self.state.last_search_results_df = filtered.copy()
        self.map_manager.update_search_layer(
            detections_geojson,
            style_callback=self._search_style_callback,
        )
        self.tile_panel.update_results(
            filtered,
            auto_show=False,
            on_ready=self._on_tiles_ready,
        )

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
        t0 = time.time()
        # Build GeoJSON from cached geometries - instant, no DB queries
        pos_features = []
        for pid in self.state.pos_ids:
            if pid in self.state.cached_geometries:
                pos_features.append(
                    {
                        "type": "Feature",
                        "geometry": self.state.cached_geometries[pid],
                        "properties": {"id": pid},
                    }
                )

        neg_features = []
        for pid in self.state.neg_ids:
            if pid in self.state.cached_geometries:
                neg_features.append(
                    {
                        "type": "Feature",
                        "geometry": self.state.cached_geometries[pid],
                        "properties": {"id": pid},
                    }
                )

        t1 = time.time()
        # Query any missing geometries (should be rare)
        missing_ids = [
            pid
            for pid in self.state.pos_ids + self.state.neg_ids
            if pid not in self.state.cached_geometries
        ]
        if missing_ids:
            print(f"[TIMING] ⚠️ Querying {len(missing_ids)} missing geometries")
            import sys

            sys.stdout.flush()
            t2 = time.time()
            df = self.data.duckdb_connection.execute(
                f"""
                SELECT id, ST_AsGeoJSON(geometry) as geometry
                FROM geo_embeddings
                WHERE id IN ({",".join(["?" for _ in missing_ids])})
                """,
                [str(pid) for pid in missing_ids],
            ).df()
            t3 = time.time()
            missing_ms = (t3 - t2) * 1000
            print(f"[TIMING] missing geometries query: {missing_ms:.1f}ms")
            sys.stdout.flush()
            for _, row in df.iterrows():
                pid = str(row["id"])
                geom = json.loads(row["geometry"])
                self.state.cached_geometries[pid] = geom
                feature = {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"id": pid},
                }
                if pid in self.state.pos_ids:
                    pos_features.append(feature)
                if pid in self.state.neg_ids:
                    neg_features.append(feature)

        t4 = time.time()
        self.map_manager.update_label_layers(
            pos_geojson={"type": "FeatureCollection", "features": pos_features},
            neg_geojson={"type": "FeatureCollection", "features": neg_features},
            erase_geojson=self._empty_collection(),
        )
        t5 = time.time()
        build_ms = (t1 - t0) * 1000
        update_ms = (t5 - t4) * 1000
        print(
            f"[TIMING] build features: {build_ms:.1f}ms, update layers: {update_ms:.1f}ms"
        )
        import sys

        sys.stdout.flush()

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
        t0 = time.time()
        if not self.state.pos_ids:
            self.state.query_vector = None
            return
        # Only fetch embeddings that aren't already cached - avoid redundant DB queries
        missing_pos = [
            pid for pid in self.state.pos_ids if pid not in self.state.cached_embeddings
        ]
        missing_neg = [
            pid for pid in self.state.neg_ids if pid not in self.state.cached_embeddings
        ]
        t1 = time.time()
        if missing_pos or missing_neg:
            print(
                f"[TIMING] ⚠️ Fetching {len(missing_pos)} pos, {len(missing_neg)} neg embeddings"
            )
            import sys

            sys.stdout.flush()
        if missing_pos:
            t2 = time.time()
            self._fetch_embeddings(missing_pos)
            t3 = time.time()
            fetch_pos_ms = (t3 - t2) * 1000
            print(f"[TIMING] fetch_pos: {fetch_pos_ms:.1f}ms")
            sys.stdout.flush()
        if missing_neg:
            t4 = time.time()
            self._fetch_embeddings(missing_neg)
            t5 = time.time()
            fetch_neg_ms = (t5 - t4) * 1000
            print(f"[TIMING] fetch_neg: {fetch_neg_ms:.1f}ms")
            sys.stdout.flush()
        t6 = time.time()
        self.state.update_query_vector()
        t7 = time.time()
        check_ms = (t1 - t0) * 1000
        update_vec_ms = (t7 - t6) * 1000
        print(
            f"[TIMING] check cache: {check_ms:.1f}ms, update_vector: {update_vec_ms:.1f}ms"
        )
        import sys

        sys.stdout.flush()

    def _on_tiles_ready(self) -> None:
        self.tiles_button.color = "success"
        self.tiles_button.outlined = False

    def _reset_tiles_button(self) -> None:
        self.tiles_button.color = None
        self.tiles_button.outlined = True

    def _display_id_from_row(self, row) -> str:
        column = getattr(self, "external_id_column", "id")
        if column != "id" and column in row.index:
            value = row[column]
            if pd.isna(value):
                return str(row["id"])
            return str(value)
        for candidate in ("source_id", "tile_id"):
            if candidate in row.index and not pd.isna(row[candidate]):
                return str(row[candidate])
        return str(row["id"])

    def _handle_tile_label(self, point_id: str, row, label: str) -> None:
        # Detection mode: label detections differently
        if self.state.detection_mode:
            tile_id = point_id  # In detection mode, point_id is the tile_id

            if label == UIConstants.POSITIVE_LABEL:
                new_label = 1
                label_name = "positive"
            else:
                new_label = 0
                label_name = "negative"

            current_label = self.state.detection_labels.get(tile_id)
            if current_label == new_label:
                # Toggle off
                del self.state.detection_labels[tile_id]
                self._show_operation_status("✅ Removed label from detection")
            else:
                self.state.label_detection(tile_id, new_label)
                num_labeled = len(self.state.detection_labels)
                self._show_operation_status(
                    f"✅ Labeled as {label_name} | Total: {num_labeled}"
                )
            self._refresh_detection_layer()
            return

        # Normal mode: fetch embeddings and apply label
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

        polygon = self._tile_polygon_from_spec(lat, lon)
        if polygon is None:
            half_size = 0.0025 / 2
            square_coords = [
                (lon - half_size, lat - half_size),
                (lon + half_size, lat - half_size),
                (lon + half_size, lat + half_size),
                (lon - half_size, lat + half_size),
                (lon - half_size, lat - half_size),
            ]
            polygon = shapely.geometry.Polygon(square_coords)

        self.map_manager.highlight_polygon(polygon, color="red", fill_opacity=0.0)
        self._show_operation_status("📍 Centered on tile")

    def _tile_polygon_from_spec(self, lat: float, lon: float):
        tile_spec = getattr(self.data, "tile_spec", None)
        if not tile_spec:
            return None

        meters_per_pixel = tile_spec.get("meters_per_pixel")
        tile_size_px = tile_spec.get("tile_size_px")
        if not meters_per_pixel or not tile_size_px:
            return None

        half_side = (meters_per_pixel * tile_size_px) / 2.0
        if half_side <= 0:
            return None

        zone = int((lon + 180) // 6) + 1
        zone = max(1, min(zone, 60))
        epsg = 32600 + zone if lat >= 0 else 32700 + zone

        try:
            utm_crs = pyproj.CRS.from_epsg(epsg)
            forward = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            inverse = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
            x, y = forward.transform(lon, lat)
            square = shapely.geometry.box(
                x - half_side,
                y - half_side,
                x + half_side,
                y + half_side,
            )
            return shapely.ops.transform(inverse.transform, square)
        except Exception:
            return None

    def _filter_detection_layer(self, threshold: float) -> None:
        if not self.state.detection_data:
            return
        features = self.state.detection_data.get("features", [])
        filtered_features = [
            f
            for f in features
            if f.get("properties", {}).get("probability", 0.0) >= threshold
        ]
        filtered_geojson = {"type": "FeatureCollection", "features": filtered_features}
        self.map_manager.update_detection_layer(
            filtered_geojson, style_callback=self._detection_style_callback
        )
        num_shown = len(filtered_features)
        num_total = len(features)
        num_labeled = len(self.state.detection_labels)
        self._show_operation_status(
            f"🔍 {num_shown}/{num_total} detections | {num_labeled} labeled"
        )

    def _refresh_detection_layer(self) -> None:
        """Refresh detection layer with current threshold and labels."""
        threshold = self.detection_threshold_slider.v_model
        self._filter_detection_layer(threshold)

    def _update_detection_tiles(self) -> None:
        """Update the tile panel with current detections, sorted by lowest probability first."""
        if not self.state.detection_mode or not self.state.detection_data:
            return

        threshold = self.detection_threshold_slider.v_model
        features = self.state.detection_data.get("features", [])

        # Filter by threshold
        filtered = [
            f
            for f in features
            if f.get("properties", {}).get("probability", 0.0) >= threshold
        ]

        if not filtered:
            self.tile_panel.clear()
            return

        # Build DataFrame for tile panel (sorted by lowest probability first)
        records = []
        for feature in filtered:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            tile_id = props.get("tile_id", props.get("id", "unknown"))
            probability = props.get("probability", 0.5)

            # Convert geometry to WKT
            geom_shape = shapely.geometry.shape(geom)
            centroid = geom_shape.centroid

            records.append(
                {
                    "id": str(tile_id),
                    "probability": probability,
                    "geometry_wkt": centroid.wkt,
                    "geometry_json": json.dumps(shapely.geometry.mapping(centroid)),
                }
            )

        df = pd.DataFrame(records)
        # Sort by probability ascending (lowest first = hardest cases)
        df = df.sort_values("probability", ascending=True).reset_index(drop=True)

        # Update tile panel
        self.tile_panel.update_results(df, auto_show=False)

    def _detection_style_callback(self, feature):
        """Style callback for detection layer that shows labeled detections in pos/neg colors."""
        from geovibes.ui_config import LayerStyles

        props = feature.get("properties", {})
        tile_id = props.get("tile_id", props.get("id", "unknown"))
        probability = props.get("probability", 0.5)

        # Check if this detection has been labeled
        label = self.state.detection_labels.get(tile_id)

        if label == 1:
            # Labeled as positive - use blue
            color = UIConstants.POS_COLOR
        elif label == 0:
            # Labeled as negative - use orange
            color = UIConstants.NEG_COLOR
        else:
            # Not labeled - normalize probability to dataset range for colormap
            min_prob = getattr(self, "_detection_prob_min", 0.0)
            max_prob = getattr(self, "_detection_prob_max", 1.0)
            if max_prob > min_prob:
                normalized = (probability - min_prob) / (max_prob - min_prob)
            else:
                normalized = 0.5
            color = LayerStyles.probability_to_color(normalized)

        return {
            "color": color,
            "weight": 3 if label is not None else 2,
            "opacity": 0.9 if label is not None else 0.8,
            "fillColor": color,
            "fillOpacity": 0.2 if label is not None else 0.1,
        }

    def _handle_save_dataset(self) -> None:
        if self.state.detection_mode:
            result = self.dataset_manager.export_augmented_dataset()
        else:
            result = self.dataset_manager.save_dataset()

        if result:
            geojson_path = result.get("geojson")
            csv_path = result.get("csv")
            if geojson_path and csv_path:
                message = f"✅ Dataset saved: {geojson_path} (labels: {csv_path})"
            elif geojson_path:
                message = f"✅ Dataset saved: {geojson_path}"
            else:
                message = "✅ Dataset saved"
            self._show_operation_status(message)
        else:
            self._show_operation_status("⚠️ Nothing to save")

    def reset_all(self, _button=None, clear_overlays: bool = False) -> None:
        if self.verbose:
            print("🗑️ Resetting all labels and search results...")
        self.state.reset()
        self.map_manager.update_label_layers(
            pos_geojson=self._empty_collection(),
            neg_geojson=self._empty_collection(),
            erase_geojson=self._empty_collection(),
        )
        self.map_manager.update_search_layer(self._empty_collection())
        self.map_manager.clear_detection_layer()
        self.map_manager.clear_vector_layer()
        self.map_manager.clear_highlight()
        if clear_overlays:
            self.map_manager.clear_overlay_layers()
        self.detection_controls.style_ = "display: none;"
        # Reset slider and colormap range to defaults
        self.detection_threshold_slider.min = 0.0
        self.detection_threshold_slider.max = 1.0
        self.detection_threshold_slider.v_model = 0.5
        self.detection_threshold_label.children = ["0.50"]
        self._detection_prob_min = 0.0
        self._detection_prob_max = 1.0
        self.tile_panel.clear()
        self.tile_panel.hide()
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

    def _update_status(
        self, lat: Optional[float] = None, lon: Optional[float] = None
    ) -> None:
        self.map_manager.update_status(lat=lat, lon=lon)

    def _show_operation_status(self, message: str) -> None:
        self.map_manager.set_operation(message)

    def _clear_operation_status(self) -> None:
        self.map_manager.clear_operation()

    @staticmethod
    def _empty_collection() -> Dict:
        return {"type": "FeatureCollection", "features": []}

    # ------------------------------------------------------------------
    # Overlay tile layer API
    # ------------------------------------------------------------------

    def add_tile_layer(
        self, url: str, name: str, opacity: float = 1.0, attribution: str = ""
    ) -> None:
        """Add an XYZ tile layer overlay."""
        self.map_manager.add_tile_layer(url, name, opacity, attribution)

    def add_ee_layer(
        self, ee_image, vis_params: Dict, name: str, opacity: float = 1.0
    ) -> None:
        """Add an Earth Engine image as a tile layer overlay."""
        self.map_manager.add_ee_layer(ee_image, vis_params, name, opacity)

    def remove_layer(self, name: str) -> bool:
        """Remove an overlay layer by name."""
        return self.map_manager.remove_layer(name)

    def set_layer_opacity(self, name: str, opacity: float) -> None:
        """Set the opacity of an overlay layer."""
        self.map_manager.set_layer_opacity(name, opacity)

    def list_layers(self) -> List[str]:
        """Return names of all overlay layers."""
        return self.map_manager.list_overlay_layers()

    def close(self) -> None:
        self.data.close()


__all__ = ["GeoVibes"]
