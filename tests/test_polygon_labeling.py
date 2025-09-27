import json
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry

from geovibes.ui.app import GeoVibes
from geovibes.ui.state import AppState


class DummyDrawControl:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyMapManager:
    def __init__(self):
        self.draw_control = DummyDrawControl()
        self.last_search_layer = None
        self.last_label_layers = None

    def update_search_layer(self, *args, **kwargs):
        self.last_search_layer = (args, kwargs)

    def update_label_layers(self, **kwargs):
        self.last_label_layers = kwargs

    def set_operation(self, message):
        self.last_operation = message

    def clear_operation(self):
        self.last_operation = None

    def update_status(self, **kwargs):
        pass


class DummyTilePanel:
    def __init__(self):
        self.received = None

    def update_results(self, df):
        self.received = df

    def clear(self):
        self.received = None

    def hide(self):
        pass


@pytest.fixture
def geo_vibes_stub():
    gv = GeoVibes.__new__(GeoVibes)
    gv.state = AppState()
    gv.map_manager = DummyMapManager()
    gv.tile_panel = DummyTilePanel()
    gv.tiles_button = SimpleNamespace(button_style="")

    # Replace heavy operations with no-ops for isolated testing
    gv._show_operation_status = lambda *args, **kwargs: None
    gv._update_status = lambda *args, **kwargs: None
    gv._fetch_embeddings = lambda *args, **kwargs: None
    gv._update_layers = lambda *args, **kwargs: None
    gv._update_query_vector = lambda *args, **kwargs: None

    return gv


def _sample_results_df():
    point = shapely.geometry.Point(-1.0, 37.0)
    return pd.DataFrame(
        {
            "id": ["123"],
            "distance": [0.42],
            "geometry_wkt": [point.wkt],
            "geometry_json": [json.dumps(shapely.geometry.mapping(point))],
        }
    )


def test_process_search_results_creates_geodataframe(geo_vibes_stub):
    gv = geo_vibes_stub
    results_df = _sample_results_df()

    gv._process_search_results(results_df, n_neighbors=1)

    detections = gv.state.detections_with_embeddings
    assert isinstance(detections, gpd.GeoDataFrame)
    assert not detections.empty
    assert "geometry" in detections

    polygon = shapely.geometry.box(-1.5, 36.5, -0.5, 37.5)
    within_mask = detections.geometry.within(polygon)
    assert within_mask.any(), "Expected cached detections to support spatial predicates"


def test_handle_draw_uses_geodataframe_geometry(geo_vibes_stub):
    gv = geo_vibes_stub
    results_df = _sample_results_df()
    gv._process_search_results(results_df, n_neighbors=1)

    polygon = shapely.geometry.box(-1.5, 36.5, -0.5, 37.5)
    geo_json = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(polygon.exterior.coords)],
        }
    }

    # Should not raise when polygon labeling intersects cached detections
    gv._handle_draw(None, "created", geo_json)

    assert "123" in gv.state.pos_ids
    assert gv.map_manager.draw_control.cleared is True
