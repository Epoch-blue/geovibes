import json

import pandas as pd
import numpy as np

from geovibes.ui.datasets import DatasetManager
from geovibes.ui.state import AppState
from geovibes.ui_config import UIConstants


class DummyData:
    duckdb_connection = None
    embedding_dim = 3


class DummyMap:
    def __init__(self):
        self.last_vector = None

    def set_vector_layer(self, geojson_data, name):
        self.last_vector = (geojson_data, name)


def test_load_from_geojson_updates_state():
    state = AppState()
    manager = DatasetManager(DummyData(), DummyMap(), state)

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "id": "123",
                    "label": UIConstants.POSITIVE_LABEL,
                    "embedding": [1.0, 0.0, 0.0],
                },
            }
        ],
    }
    manager.load_from_content(json.dumps(payload).encode("utf-8"), "labels.geojson")

    assert state.pos_ids == ["123"]
    np.testing.assert_array_equal(state.cached_embeddings["123"], np.array([1.0, 0.0, 0.0]))


def test_add_vector_layer_from_geojson():
    state = AppState()
    dummy_map = DummyMap()
    manager = DatasetManager(DummyData(), dummy_map, state)

    geojson = {"type": "FeatureCollection", "features": []}
    manager.add_vector_from_content(
        json.dumps(geojson).encode("utf-8"), "example.geojson"
    )

    assert dummy_map.last_vector[1] == "vector_layer_example.geojson"
    assert dummy_map.last_vector[0]["type"] == "FeatureCollection"


def test_save_dataset_persists_geojson(tmp_path, monkeypatch):
    state = AppState()
    state.pos_ids = ["123"]
    state.neg_ids = ["456"]
    state.cached_embeddings["123"] = np.array([0.1, 0.2, 0.3])

    df = pd.DataFrame(
        [
            {
                "id": "123",
                "geometry_json": json.dumps({"type": "Point", "coordinates": [1.0, 2.0]}),
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "id": "456",
                "geometry_json": json.dumps({"type": "Point", "coordinates": [3.0, 4.0]}),
                "embedding": [0.7, 0.8, 0.9],
            },
        ]
    )

    class FakeResult:
        def __init__(self, frame):
            self._frame = frame

        def df(self):
            return self._frame

    class FakeConnection:
        def __init__(self, frame):
            self._frame = frame
            self.last_query = None
            self.last_params = None

        def execute(self, query, params):
            self.last_query = query
            self.last_params = params
            return FakeResult(self._frame)

    class FrozenDateTime:
        @classmethod
        def now(cls):
            class FrozenNow:
                def strftime(self, _fmt):
                    return "20250101_010203"

            return FrozenNow()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("geovibes.ui.datasets.datetime", FrozenDateTime)

    dummy_data = DummyData()
    dummy_data.duckdb_connection = FakeConnection(df)
    dummy_data.embedding_dim = 3

    manager = DatasetManager(dummy_data, DummyMap(), state)
    result = manager.save_dataset()

    assert set(dummy_data.duckdb_connection.last_params) == {"123", "456"}
    assert result["geojson"] == "labeled_dataset_20250101_010203.geojson"
    assert result["positive"] == "1"
    assert result["negative"] == "1"
    assert "csv" not in result

    geojson_path = tmp_path / result["geojson"]
    assert geojson_path.exists()

    with geojson_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["metadata"]["total_points"] == 2
    assert payload["metadata"]["positive_points"] == 1
    assert payload["metadata"]["negative_points"] == 1

    features_by_id = {feat["properties"]["id"]: feat for feat in payload["features"]}

    assert features_by_id["123"]["properties"]["label"] == UIConstants.POSITIVE_LABEL
    assert features_by_id["123"]["properties"]["embedding"] == [0.1, 0.2, 0.3]

    assert features_by_id["456"]["properties"]["label"] == UIConstants.NEGATIVE_LABEL
    assert features_by_id["456"]["properties"]["embedding"] == [0.7, 0.8, 0.9]
