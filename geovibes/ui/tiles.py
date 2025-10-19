"""Tile panel for displaying search results as thumbnails."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

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
                overflow_y="visible",
                flex="1 1 auto",
                height="auto",
            ),
        )

        self.container = VBox(
            [controls, self.results_grid],
            layout=Layout(
                display="none",
                width="265px",
                padding="5px",
                height=UIConstants.DEFAULT_HEIGHT,
                overflow="auto",
            ),
        )

        self.control = self.map_manager.add_widget_control(
            self.container, position="topright"
        )

        self.results_ready = False
        self._tiles_ready_callback: Optional[Callable[[], None]] = None
        self._page_sizes: List[int] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self._pending_futures: set[concurrent.futures.Future] = set()
        self._pending_batches: Dict[object, Dict[str, Any]] = {}
        self._loader_token: Optional[object] = None
        try:
            loop = asyncio.get_event_loop()
            self._async_loop = loop if loop.is_running() else None
        except RuntimeError:
            self._async_loop = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self.container.layout.display == "none":
            self.container.layout.display = "block"
            if (
                not self.results_ready
                and self.state.last_search_results_df is not None
                and not self.results_grid.children
            ):
                self._render_current_page(
                    append=False,
                    on_finish=self._handle_tiles_ready,
                    loader_token=self._loader_token,
                )
        else:
            self.container.layout.display = "none"

    def show(self) -> None:
        self.container.layout.display = "block"

    def hide(self) -> None:
        self.container.layout.display = "none"

    def clear(self) -> None:
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self.results_grid.children = []
        self.next_tiles_btn.layout.display = "none"
        self.state.last_search_results_df = None
        self.state.tile_page = 0
        self._update_operation(None)
        self.results_ready = False
        self._tiles_ready_callback = None
        self._page_sizes = []
        self._loader_token = None

    def update_results(
        self,
        search_results_df,
        *,
        auto_show: bool = False,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        for future in list(self._pending_futures):
            future.cancel()
        self._pending_futures.clear()
        self._pending_batches.clear()
        self._tiles_ready_callback = on_ready
        self.results_ready = False
        self._page_sizes = []
        self.state.last_search_results_df = search_results_df
        self.state.tile_page = 0
        self.results_grid.children = []
        self.next_tiles_btn.layout.display = "none"
        token = object()
        self._loader_token = token
        if auto_show:
            self.show()
        self._render_current_page(
            append=False, on_finish=self._handle_tiles_ready, loader_token=token
        )

    def reload_tiles_for_new_basemap(self) -> None:
        if self.state.last_search_results_df is None:
            return
        current_count = len(self.results_grid.children)
        if current_count == 0:
            return
        self._render_current_page(
            append=False,
            limit=current_count,
            loader_token=self._loader_token,
        )

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
        next_size = 10
        self.state.tile_page += 1
        self._render_current_page(
            append=True,
            page_size=next_size,
            loader_token=self._loader_token,
        )

    def handle_map_basemap_change(self, basemap_name: str) -> None:
        if basemap_name in self.allowed_sources:
            if self.tile_basemap_dropdown.value != basemap_name:
                self.tile_basemap_dropdown.value = basemap_name
            else:
                self.state.tile_basemap = basemap_name
                self.reload_tiles_for_new_basemap()
        elif self.tile_basemap_dropdown.value not in self.allowed_sources:
            self.tile_basemap_dropdown.value = self.allowed_sources[0]

    def _render_current_page(
        self,
        append: bool,
        limit: Optional[int] = None,
        on_finish: Optional[Callable[[], None]] = None,
        page_size: Optional[int] = None,
        loader_token: Optional[object] = None,
    ) -> None:
        df = self.state.last_search_results_df
        if df is None or df.empty:
            self.results_grid.children = []
            self.next_tiles_btn.layout.display = "none"
            self._update_operation(None)
            if on_finish:
                on_finish()
            return

        refresh_only = limit is not None and not append

        if refresh_only:
            start_index = 0
            end_index = min(limit, len(df))
        else:
            already_loaded = sum(self._page_sizes)
            default_size = (
                self.state.tiles_per_page
                if self._page_sizes
                else self.state.initial_load_size
            )

            desired = page_size if page_size is not None else default_size
            desired = max(1, desired)

            start_index = already_loaded if append or self._page_sizes else 0
            end_index = min(start_index + desired, len(df))

        page_df = df.iloc[start_index:end_index]
        if page_df.empty:
            self.next_tiles_btn.layout.display = "none"
            if on_finish:
                on_finish()
            return

        placeholder_widgets = [
            self._make_placeholder_tile() for _ in range(len(page_df))
        ]
        current_children = list(self.results_grid.children)
        if append:
            placeholder_start = len(current_children)
            current_children.extend(placeholder_widgets)
        else:
            placeholder_start = 0
            current_children = placeholder_widgets
        self.results_grid.children = tuple(current_children)

        placeholder_indices = [
            placeholder_start + idx for idx in range(len(placeholder_widgets))
        ]

        batch_token = loader_token if loader_token is not None else object()
        remaining = len(page_df)
        batch_info = {
            "remaining": remaining,
            "on_finish": on_finish if not append else None,
            "token": loader_token,
            "update_status": "✅ Tiles updated",
        }
        self._pending_batches[batch_token] = batch_info

        if not refresh_only:
            if not append:
                self._page_sizes = [remaining]
            else:
                self._page_sizes.append(remaining)

        self._update_operation("⏳ Loading tiles...")

        for offset, (_, row) in enumerate(page_df.iterrows()):
            target_index = placeholder_indices[offset]
            future = self._executor.submit(self._create_tile_widget, row, append)
            self._pending_futures.add(future)
            future.add_done_callback(
                lambda fut, idx=target_index, bt=batch_token: self._on_tile_future_done(
                    fut, idx, bt
                )
            )

        if end_index < len(df):
            self.next_tiles_btn.layout.display = "flex"
        else:
            self.next_tiles_btn.layout.display = "none"

    def _handle_tiles_ready(self) -> None:
        if self.results_ready:
            return
        self.results_ready = True
        callback = self._tiles_ready_callback
        self._tiles_ready_callback = None
        if callback:
            callback()

    def _dispatch_to_ui(self, func: Callable[..., None], *args) -> None:
        if self._async_loop and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(func, *args)
        else:
            func(*args)

    def _on_tile_future_done(
        self,
        future: concurrent.futures.Future,
        target_index: int,
        batch_token: object,
    ) -> None:
        self._pending_futures.discard(future)
        batch = self._pending_batches.get(batch_token)
        if batch is None:
            return
        token = batch.get("token")
        if future.cancelled():
            self._pending_batches.pop(batch_token, None)
            return
        try:
            widget = future.result()
        except Exception:
            widget = self._make_placeholder_tile()

        def apply_result() -> None:
            current_batch = self._pending_batches.get(batch_token)
            if current_batch is None:
                return
            if token is not None and token is not self._loader_token:
                self._pending_batches.pop(batch_token, None)
                return
            current_children = list(self.results_grid.children)
            if target_index < len(current_children):
                current_children[target_index] = widget
                self.results_grid.children = tuple(current_children)
            current_batch["remaining"] -= 1
            if current_batch["remaining"] <= 0:
                self._pending_batches.pop(batch_token, None)
                if current_batch.get("update_status"):
                    self._update_operation(current_batch["update_status"])
                finish_cb = current_batch.get("on_finish")
                if finish_cb and (token is None or token is self._loader_token):
                    finish_cb()

        self._dispatch_to_ui(apply_result)

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
            tile_spec = getattr(
                getattr(self.map_manager, "data", None), "tile_spec", None
            )
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

    def _handle_tiles_ready(self) -> None:
        if self.results_ready:
            return
        self.results_ready = True
        callback = self._tiles_ready_callback
        self._tiles_ready_callback = None
        if callback:
            callback()


__all__ = ["TilePanel"]
