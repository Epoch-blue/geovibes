"""Tile panel for displaying search results as thumbnails."""

from __future__ import annotations

import concurrent.futures
from functools import partial
from typing import Callable, Optional

import ipywidgets as ipyw
import shapely.wkt
from ipywidgets import Button, GridBox, HBox, Image, Label, Layout, VBox

from geovibes.ui_config import BasemapConfig, UIConstants
from .xyz import get_map_image

TILE_SOURCES = ("HUTCH_TILE", "MAPTILER", "GOOGLE_HYBRID")


class TilePanel:
    """Manages the tile-based results pane."""

    def __init__(
        self,
        *,
        state,
        map_manager,
        on_label: Callable[[str, dict, str], None],
        on_center: Callable[[dict], None],
        verbose: bool = False,
    ) -> None:
        self.state = state
        self.map_manager = map_manager
        self.on_label = on_label
        self.on_center = on_center
        self.verbose = verbose

        self.allowed_sources = [
            name for name in TILE_SOURCES if name in BasemapConfig.BASEMAP_TILES
        ]
        if not self.allowed_sources:
            self.allowed_sources = list(BasemapConfig.BASEMAP_TILES.keys())

        if self.state.tile_basemap not in self.allowed_sources:
            self.state.tile_basemap = self.allowed_sources[0]

        self.tile_basemap_dropdown = ipyw.Dropdown(
            options=self.allowed_sources,
            value=self.state.tile_basemap,
            description="",
            layout=Layout(width="180px"),
            style={"description_width": "initial"},
        )
        self.tile_basemap_dropdown.observe(self._on_tile_basemap_change, names="value")

        self.next_tiles_btn = Button(
            description="Next",
            layout=Layout(width="60px", margin="0 0 0 5px", display="none"),
        )
        self.next_tiles_btn.on_click(self._on_next_tiles_click)

        controls = HBox(
            [self.tile_basemap_dropdown, self.next_tiles_btn],
            layout=Layout(align_items="center", margin="0 0 10px 0"),
        )

        self.results_grid = GridBox(
            [],
            layout=Layout(
                width="100%",
                grid_template_columns="1fr 1fr",
                grid_gap="3px",
                overflow_y="auto",
                flex="1 1 auto",
                height="100%",
            ),
        )

        self.container = VBox(
            [controls, self.results_grid],
            layout=Layout(
                display="none",
                width="265px",
                padding="5px",
                height=UIConstants.DEFAULT_HEIGHT,
                overflow="hidden",
            ),
        )

        self.control = self.map_manager.add_widget_control(self.container, position="topright")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self.container.layout.display == "none":
            self.container.layout.display = "block"
        else:
            self.container.layout.display = "none"

    def show(self) -> None:
        self.container.layout.display = "block"

    def hide(self) -> None:
        self.container.layout.display = "none"

    def clear(self) -> None:
        self.results_grid.children = []
        self.next_tiles_btn.layout.display = "none"
        self.state.last_search_results_df = None
        self.state.tile_page = 0
        self._update_operation(None)

    def update_results(self, search_results_df) -> None:
        self.state.last_search_results_df = search_results_df
        self.state.tile_page = 0
        self.results_grid.children = []
        self.show()
        self._render_current_page(append=False)

    def reload_tiles_for_new_basemap(self) -> None:
        if self.state.last_search_results_df is None:
            return
        current_count = len(self.results_grid.children)
        if current_count == 0:
            return
        self._render_current_page(append=False, limit=current_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_tile_basemap_change(self, change) -> None:
        new_value = change["new"]
        if new_value not in self.allowed_sources:
            return
        self.state.tile_basemap = new_value
        self.reload_tiles_for_new_basemap()

    def _on_next_tiles_click(self, _button) -> None:
        self.state.tile_page += 1
        self._render_current_page(append=True)

    def handle_map_basemap_change(self, basemap_name: str) -> None:
        if basemap_name in self.allowed_sources:
            if self.tile_basemap_dropdown.value != basemap_name:
                self.tile_basemap_dropdown.value = basemap_name
            else:
                self.state.tile_basemap = basemap_name
                self.reload_tiles_for_new_basemap()
        elif self.tile_basemap_dropdown.value not in self.allowed_sources:
            self.tile_basemap_dropdown.value = self.allowed_sources[0]

    def _render_current_page(self, append: bool, limit: Optional[int] = None) -> None:
        df = self.state.last_search_results_df
        if df is None or df.empty:
            self.results_grid.children = []
            self.next_tiles_btn.layout.display = "none"
            self._update_operation(None)
            return

        if append:
            start_index = (
                self.state.initial_load_size
                + (self.state.tile_page - 1) * self.state.tiles_per_page
            )
            end_index = start_index + self.state.tiles_per_page
        else:
            start_index = 0
            if limit is not None:
                end_index = limit
            elif self.state.tile_page == 0:
                end_index = self.state.initial_load_size
            else:
                end_index = (
                    self.state.initial_load_size
                    + self.state.tile_page * self.state.tiles_per_page
                )

        page_df = df.iloc[start_index:end_index]
        if page_df.empty:
            self.next_tiles_btn.layout.display = "none"
            self._update_operation(None)
            return

        placeholder_widgets = [self._make_placeholder_tile() for _ in range(len(page_df))]
        current_children = list(self.results_grid.children)
        if append:
            placeholder_start = len(current_children)
            current_children.extend(placeholder_widgets)
        else:
            placeholder_start = 0
            current_children = placeholder_widgets
        self.results_grid.children = tuple(current_children)

        placeholder_indices = [placeholder_start + idx for idx in range(len(placeholder_widgets))]

        self._update_operation("⏳ Loading tiles...")

        try:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for offset, (_, row) in enumerate(page_df.iterrows()):
                    target_index = placeholder_indices[offset]
                    futures.append(
                        (target_index, executor.submit(self._create_tile_widget, row, append))
                    )

                for target_index, future in futures:
                    tile_widget = future.result()
                    current_children = list(self.results_grid.children)
                    if target_index < len(current_children):
                        current_children[target_index] = tile_widget
                        self.results_grid.children = tuple(current_children)
        finally:
            self._update_operation("✅ Tiles updated")

        if end_index < len(df):
            self.next_tiles_btn.layout.display = "flex"
        else:
            self.next_tiles_btn.layout.display = "none"

    def _create_tile_widget(self, row, append: bool):
        geom = shapely.wkt.loads(row["geometry_wkt"])
        display_value = row.get("source_id")
        if display_value is None or display_value != display_value:
            display_value = row.get("tile_id")
        if display_value is None or display_value != display_value:
            display_value = row["id"]
        display_id = str(display_value)
        base_image_layout = {
            "width": "115px",
            "height": "115px",
            "overflow": "hidden",
        }
        image_layout = Layout(
            width="115px",
            height="115px",
            overflow="hidden",
        )
        try:
            tile_spec = getattr(getattr(self.map_manager, "data", None), "tile_spec", None)
            image_bytes = get_map_image(
                source=self.state.tile_basemap,
                lon=geom.x,
                lat=geom.y,
                tile_spec=tile_spec,
            )
            tile_image = Image(
                value=image_bytes,
                format="png",
                width=115,
                height=115,
                layout=image_layout,
            )
        except Exception:
            tile_image = Label(
                value="Image unavailable",
                layout=Layout(
                    width=base_image_layout["width"],
                    height=base_image_layout["height"],
                    overflow=base_image_layout["overflow"],
                    border="1px solid #ccc",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                ),
            )

        point_id = str(row["id"])

        map_button = Button(
            icon="fa-map-marker",
            layout=Layout(width="35px", height="28px", margin="0px 2px", padding="2px"),
            tooltip=f"Center map ({display_id})",
        )
        map_button.on_click(lambda _b, r=row: self.on_center(r))

        tick_button = Button(
            icon="fa-check",
            layout=Layout(width="35px", height="28px", margin="0px 2px", padding="2px"),
            tooltip=f"Label as positive ({display_id})",
        )
        cross_button = Button(
            icon="fa-times",
            layout=Layout(width="35px", height="28px", margin="0px 2px", padding="2px"),
            tooltip=f"Label as negative ({display_id})",
        )

        self._apply_label_style(point_id, tick_button, cross_button)

        tick_button.on_click(
            lambda _b, pid=point_id, r=row: self._handle_label_click(
                pid, r, UIConstants.POSITIVE_LABEL, tick_button, cross_button
            )
        )
        cross_button.on_click(
            lambda _b, pid=point_id, r=row: self._handle_label_click(
                pid, r, UIConstants.NEGATIVE_LABEL, tick_button, cross_button
            )
        )

        button_row = HBox(
            [map_button, tick_button, cross_button],
            layout=Layout(
                justify_content="center",
                width="120px",
                height="32px",
                overflow="hidden",
            ),
        )

        return VBox(
            [button_row, tile_image],
            layout=Layout(
                border="1px solid #ccc",
                padding="2px",
                width="120px",
                height="155px",
                overflow="hidden",
            ),
        )

    def _make_placeholder_tile(self) -> VBox:
        message = Label(
            value="Loading...",
            layout=Layout(
                width="115px",
                height="115px",
                border="1px solid #ccc",
                display="flex",
                align_items="center",
                justify_content="center",
                overflow="hidden",
            ),
        )
        spacer = HBox(layout=Layout(height="32px", width="120px", overflow="hidden"))
        return VBox(
            [spacer, message],
            layout=Layout(
                border="1px solid #ccc",
                padding="2px",
                width="120px",
                height="155px",
                overflow="hidden",
            ),
        )

    def _update_operation(self, message: Optional[str]) -> None:
        if message:
            if hasattr(self.map_manager, "set_operation"):
                self.map_manager.set_operation(message)
        else:
            if hasattr(self.map_manager, "clear_operation"):
                self.map_manager.clear_operation()

    def _handle_label_click(
        self,
        point_id: str,
        row,
        label: str,
        tick_button: Button,
        cross_button: Button,
    ) -> None:
        self.on_label(point_id, row, label)
        self._apply_label_style(point_id, tick_button, cross_button)

    def _apply_label_style(
        self,
        point_id: str,
        tick_button: Button,
        cross_button: Button,
    ) -> None:
        if point_id in self.state.pos_ids:
            tick_button.button_style = "success"
            tick_button.layout.opacity = "1.0"
        else:
            tick_button.button_style = ""
            tick_button.layout.opacity = "0.3"

        if point_id in self.state.neg_ids:
            cross_button.button_style = "danger"
            cross_button.layout.opacity = "1.0"
        else:
            cross_button.button_style = ""
            cross_button.layout.opacity = "0.3"


__all__ = ["TilePanel"]
