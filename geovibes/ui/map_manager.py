"""Map construction and layer management for GeoVibes."""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

import geopandas as gpd
import ipyleaflet as ipyl
import ipywidgets as ipyw
import shapely.geometry
import shapely.geometry.base
from ipyleaflet import DrawControl, Map
from ipywidgets import HTML, HBox, Layout, VBox
from tqdm import tqdm

from geovibes.ee_tools import (
    get_ee_image_url,
    get_s2_hsv_median,
    get_s2_ndvi_median,
    get_s2_ndwi_median,
    get_s2_rgb_median,
)
from geovibes.ui_config import BasemapConfig, LayerStyles, UIConstants

from .status import StatusBus


class MapManager:
    """Responsible for the ipyleaflet map, layers, and status widgets."""

    def __init__(
        self,
        *,
        data_manager,
        state,
        status_bus: StatusBus,
        verbose: bool = False,
    ) -> None:
        self.data = data_manager
        self.state = state
        self.status_bus = status_bus
        self.verbose = verbose

        self.basemap_tiles = self._setup_basemap_tiles()
        self.current_basemap = "MAPTILER"
        basemap_url = self.basemap_tiles[self.current_basemap]
        self.basemap_layer = ipyl.TileLayer(
            url=basemap_url,
            no_wrap=True,
            name="basemap",
            attribution="",
        )

        self.map = self._build_map(
            center=(self.data.center_y, self.data.center_x)
        )
        self.legend = self._build_legend()
        self.status_bar = HTML(value="Ready")
        self.vector_layer = None
        self.highlight_layer = None

        self._add_map_layers()
        self.draw_control = self._setup_draw_control()
        self.update_status()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _setup_basemap_tiles(self) -> Dict[str, str]:
        basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()
        if self.data.ee_available:
            try:
                if self.verbose:
                    print(
                        "🛰️ Setting up Earth Engine basemaps (S2 RGB, NDVI, NDWI, HSV)..."
                    )

                boundary = self.data.ee_boundary
                start = self.data.config.start_date
                end = self.data.config.end_date

                basemap_tasks = [
                    ("S2_RGB", lambda: get_s2_rgb_median(boundary, start, end), BasemapConfig.S2_RGB_VIS_PARAMS),
                    ("S2_NDVI", lambda: get_s2_ndvi_median(boundary, start, end), BasemapConfig.NDVI_VIS_PARAMS),
                    ("S2_NDWI", lambda: get_s2_ndwi_median(boundary, start, end), BasemapConfig.NDWI_VIS_PARAMS),
                    ("S2_HSV", lambda: get_s2_hsv_median(boundary, start, end), BasemapConfig.S2_HSV_VIS_PARAMS),
                ]

                for name, image_func, vis_params in tqdm(basemap_tasks, desc="Loading Earth Engine basemaps"):
                    image = image_func()
                    basemap_tiles[name] = get_ee_image_url(image, vis_params)

                if self.verbose:
                    print("✅ Earth Engine basemaps added successfully!")
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Failed to create Earth Engine basemaps: {exc}")
                    print("⚠️  Continuing with basic basemaps only")
        elif self.verbose:
            print("⚠️  Earth Engine not available - S2/NDVI/NDWI basemaps skipped")
        return basemap_tiles

    def _build_map(self, center) -> Map:
        map_widget = Map(
            basemap=self.basemap_layer,
            center=center,
            zoom=UIConstants.DEFAULT_ZOOM,
            layout=Layout(flex="1 1 auto", height="100%"),
            scroll_wheel_zoom=True,
            attribution_control=False,
        )
        attribution_control = ipyl.AttributionControl(
            position="bottomleft",
            prefix='<a href="https://leafletjs.com">Leaflet</a> | '
            + BasemapConfig.MAPTILER_ATTRIBUTION,
        )
        map_widget.add_control(attribution_control)
        return map_widget

    @staticmethod
    def _build_legend() -> HTML:
        return HTML(
            value=f"""
            <div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>
                <div><strong>Labels:</strong> 
                    <span style='color: {UIConstants.POS_COLOR}; font-weight: bold;'>🔵 Positive</span> | 
                    <span style='color: {UIConstants.NEG_COLOR}; font-weight: bold;'>🟠 Negative</span>
                </div>
                <div style='margin-top: 3px;'><strong>Search Results:</strong> 
                    <span style='color: #00ff00; font-weight: bold;'>🟢 Most Similar</span> → 
                    <span style='color: #ffff00; font-weight: bold;'>🟡 Medium</span> → 
                    <span style='color: #ff4444; font-weight: bold;'>🔴 Least Similar</span>
                </div>
            </div>
            """
        )

    def _add_map_layers(self) -> None:
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.POS_COLOR),
        )
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.NEG_COLOR),
        )
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_erase_style(),
        )
        self.points_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=["geometry"]).to_json()),
            point_style=LayerStyles.get_search_style(),
            hover_style=LayerStyles.get_search_hover_style(),
        )

        for layer in [self.pos_layer, self.neg_layer, self.erase_layer, self.points_layer]:
            self.map.add_layer(layer)

    def _setup_draw_control(self) -> DrawControl:
        draw_control = DrawControl(
            polygon=LayerStyles.get_draw_options(),
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        draw_control.clear()
        self.map.add_control(draw_control)
        return draw_control

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_draw_handler(self, handler: Callable) -> None:
        self.draw_control.on_draw(handler)

    def make_layout(self, side_panel: ipyw.Widget) -> ipyw.Widget:
        map_with_overlays = VBox(
            [
                self.map,
                HBox(
                    [self.legend, self.status_bar],
                    layout=Layout(justify_content="space-between", padding="5px"),
                ),
            ],
            layout=Layout(flex="1 1 auto"),
        )
        return HBox(
            [side_panel, map_with_overlays],
            layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width="100%"),
        )

    def update_status(self, lat: Optional[float] = None, lon: Optional[float] = None) -> None:
        if lat is None or lon is None:
            center = self.map.center
            lat, lon = center[0], center[1]
        mode = "Polygon" if self.state.lasso_mode else "Point"
        self.status_bar.value = self.status_bus.render(
            lat=lat,
            lon=lon,
            mode=mode,
            label=self.state.current_label,
            polygon_drawing=self.state.polygon_drawing,
        )

    def set_operation(self, message: str) -> None:
        self.status_bus.set_operation(message)
        self.update_status()

    def clear_operation(self) -> None:
        self.status_bus.clear_operation()
        self.update_status()

    def update_basemap(self, basemap_name: str) -> None:
        if basemap_name not in self.basemap_tiles:
            raise ValueError(f"Unknown basemap: {basemap_name}")
        self.current_basemap = basemap_name
        self.basemap_layer.url = self.basemap_tiles[basemap_name]

    def add_widget_control(self, widget: ipyw.Widget, position: str = "topright") -> ipyl.WidgetControl:
        control = ipyl.WidgetControl(widget=widget, position=position)
        self.map.add_control(control)
        return control

    def update_boundary_layer(self, path: Optional[str]) -> None:
        for layer in list(self.map.layers):
            if getattr(layer, "name", None) == "region":
                self.map.remove_layer(layer)
        if not path:
            return
        try:
            boundary_gdf = gpd.read_file(path)
            boundary_geojson = json.loads(boundary_gdf.to_json())
            region_layer = ipyl.GeoJSON(
                name="region",
                data=boundary_geojson,
                style=LayerStyles.get_region_style(),
            )
            self.map.add_layer(region_layer)
            if self.verbose:
                print("✅ Boundary layer added successfully")
        except Exception as exc:
            if self.verbose:
                print(f"❌ Could not add boundary layer: {exc}")

    def set_vector_layer(self, geojson_data: dict, name: str, style: Optional[dict] = None) -> None:
        if self.vector_layer and self.vector_layer in self.map.layers:
            self.map.remove_layer(self.vector_layer)
        style = style or {
            "color": "#FF6B6B",
            "weight": 2,
            "opacity": 0.8,
            "fillColor": "#FF6B6B",
            "fillOpacity": 0.3,
        }
        self.vector_layer = ipyl.GeoJSON(name=name, data=geojson_data, style=style)
        self.map.add_layer(self.vector_layer)

    def clear_vector_layer(self) -> None:
        if self.vector_layer and self.vector_layer in self.map.layers:
            self.map.remove_layer(self.vector_layer)
        self.vector_layer = None

    def highlight_polygon(self, polygon: shapely.geometry.base.BaseGeometry, *, color: str = "yellow"):
        if self.highlight_layer and self.highlight_layer in self.map.layers:
            self.map.remove_layer(self.highlight_layer)
        self.highlight_layer = ipyl.GeoJSON(
            data={
                "type": "Feature",
                "geometry": shapely.geometry.mapping(polygon),
            },
            name="tile_highlight",
            style={"color": color, "fillOpacity": 0.1, "weight": 3},
        )
        self.map.add_layer(self.highlight_layer)

    def clear_highlight(self) -> None:
        if self.highlight_layer and self.highlight_layer in self.map.layers:
            self.map.remove_layer(self.highlight_layer)
        self.highlight_layer = None

    def update_label_layers(
        self,
        *,
        pos_geojson: dict,
        neg_geojson: dict,
        erase_geojson: dict,
    ) -> None:
        self.pos_layer.data = pos_geojson
        self.neg_layer.data = neg_geojson
        self.erase_layer.data = erase_geojson

    def update_search_layer(
        self,
        geojson_data: dict,
        style_callback: Optional[Callable] = None,
    ) -> None:
        self.points_layer.data = geojson_data
        if style_callback:
            self.points_layer.style_callback = style_callback

    def center_on(self, lat: float, lon: float, zoom: Optional[int] = None) -> None:
        self.map.center = (lat, lon)
        if zoom is not None:
            self.map.zoom = zoom


__all__ = ["MapManager"]
