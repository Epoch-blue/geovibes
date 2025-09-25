import json

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
