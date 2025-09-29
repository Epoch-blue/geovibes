import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
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

    def update_results(self, df, **kwargs):
        self.received = df
        self.kwargs = kwargs
        callback = kwargs.get("on_ready")
        if callback:
            callback()

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


class DummyDataManager:
    def __init__(self, results):
        self.results = list(results)
        self.index = 0
        self.duckdb_connection = SimpleNamespace(
            execute=lambda *args, **kwargs: SimpleNamespace(df=lambda: pd.DataFrame())
        )

    def nearest_point(self, lon, lat):  # pragma: no cover - trivial accessor
        if self.index >= len(self.results):
            raise AssertionError("No more nearest_point results configured")
        result = self.results[self.index]
        self.index += 1
        return result

    def fetch_embeddings(self, point_ids):  # pragma: no cover - unused helper
        data = pd.DataFrame(
            {
                "id": [str(pid) for pid in point_ids],
                "embedding": [[0.0, 0.0] for _ in point_ids],
            }
        )
        yield data


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


def test_polygon_mode_disables_point_labeling(geo_vibes_stub):
    gv = geo_vibes_stub
    called = {"value": False}

    def fake_label_point(*args, **kwargs):  # pragma: no cover - simple flag setter
        called["value"] = True

    gv.label_point = fake_label_point

    gv._on_selection_mode_change({"new": "polygon"})
    gv._on_map_interaction(type="click", coordinates=(37.0, -1.0), modifiers={})
    assert called["value"] is False

    gv._on_selection_mode_change({"new": "point"})
    gv._on_map_interaction(type="click", coordinates=(37.0, -1.0), modifiers={})
    assert called["value"] is True


def test_iterative_label_workflow_positive_then_negative(geo_vibes_stub):
    gv = geo_vibes_stub
    gv.data = DummyDataManager(
        [
            ("1", "POINT (-1 37)", 0.5, [0.1, 0.2, 0.3]),
            ("2", "POINT (-1.1 37.1)", 0.7, [0.4, 0.5, 0.6]),
        ]
    )

    layer_calls = []
    gv._update_layers = lambda: layer_calls.append(
        (list(gv.state.pos_ids), list(gv.state.neg_ids))
    )
    query_calls = []
    gv._update_query_vector = lambda: query_calls.append(tuple(gv.state.pos_ids))

    gv.label_point(lon=-1.0, lat=37.0)

    assert gv.state.pos_ids == ["1"]
    assert gv.state.neg_ids == []
    np.testing.assert_array_equal(gv.state.cached_embeddings["1"], np.array([0.1, 0.2, 0.3]))
    assert layer_calls[-1] == (["1"], [])
    assert query_calls[-1] == ("1",)

    gv._on_label_change({"new": "Negative"})
    assert gv.state.current_label == "Negative"

    gv.label_point(lon=-1.2, lat=37.2)

    assert gv.state.pos_ids == ["1"]
    assert gv.state.neg_ids == ["2"]
    assert layer_calls[-1] == (["1"], ["2"])
    assert query_calls[-1] == ("1",)


def test_relabel_point_from_positive_to_negative(geo_vibes_stub):
    gv = geo_vibes_stub
    gv.data = DummyDataManager(
        [
            ("7", "POINT (-1 37)", 0.5, [0.0, 1.0, 2.0]),
            ("7", "POINT (-1 37)", 0.4, [0.0, 1.0, 2.0]),
        ]
    )

    gv._update_layers = lambda: None
    gv._update_query_vector = lambda: None

    gv.label_point(lon=-1.0, lat=37.0)
    assert gv.state.pos_ids == ["7"]
    assert gv.state.neg_ids == []

    gv._on_label_change({"new": "Negative"})
    gv.label_point(lon=-1.0, lat=37.0)

    assert gv.state.pos_ids == []
    assert gv.state.neg_ids == ["7"]
