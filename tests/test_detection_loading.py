import numpy as np
import pytest

from geovibes.ui.datasets import DatasetManager, detect_geojson_type
from geovibes.ui.state import AppState
from geovibes.ui_config import UIConstants


class DummyDataManager:
    duckdb_connection = None
    embedding_dim = 3


class DummyMapManager:
    def __init__(self):
        self.last_vector = None
        self.last_detection_layer = None

    def set_vector_layer(self, geojson_data, name):
        self.last_vector = (geojson_data, name)

    def set_detection_layer(self, geojson_data, style_callback=None):
        self.last_detection_layer = (geojson_data, style_callback)

    def update_detection_layer(self, geojson_data, style_callback=None):
        self.last_detection_layer = (geojson_data, style_callback)


@pytest.fixture
def sample_labeled_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {
                    "id": "123",
                    "label": UIConstants.POSITIVE_LABEL,
                    "embedding": [0.1, 0.2, 0.3],
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                "properties": {
                    "id": "456",
                    "label": UIConstants.NEGATIVE_LABEL,
                    "embedding": [0.4, 0.5, 0.6],
                },
            },
        ],
    }


@pytest.fixture
def sample_detection_geojson():
    return {
        "type": "FeatureCollection",
        "name": "classification_detections",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 100744,
                    "tile_id": "16RCA_32_16_10_171_398",
                    "probability": 0.54936039447784424,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-88.439470812586592, 31.872439676299138],
                            [-88.43951569212075, 31.875325799841779],
                            [-88.442897891666135, 31.87528745524277],
                            [-88.442852906774789, 31.872401335983934],
                            [-88.439470812586592, 31.872439676299138],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": 100745,
                    "tile_id": "16RCA_32_16_10_172_398",
                    "probability": 0.68626922369003296,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-88.439493251549493, 31.873882738241495],
                            [-88.439538134300506, 31.876768861099961],
                            [-88.442920386530147, 31.87673051435895],
                            [-88.442875398414415, 31.873844395784431],
                            [-88.439493251549493, 31.873882738241495],
                        ]
                    ],
                },
            },
        ],
    }


@pytest.fixture
def sample_vector_layer_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                "properties": {"name": "Test Point", "category": "marker"},
            }
        ],
    }


def test_detect_geojson_type_labeled(sample_labeled_geojson):
    result = detect_geojson_type(sample_labeled_geojson)
    assert result == "labeled"


def test_detect_geojson_type_detections(sample_detection_geojson):
    result = detect_geojson_type(sample_detection_geojson)
    assert result == "detections"


def test_detect_geojson_type_vector_layer(sample_vector_layer_geojson):
    result = detect_geojson_type(sample_vector_layer_geojson)
    assert result == "vector_layer"


def test_detect_geojson_type_empty_features():
    geojson = {"type": "FeatureCollection", "features": []}
    result = detect_geojson_type(geojson)
    assert result == "vector_layer"


def test_detect_geojson_type_no_label_but_has_embedding():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"id": "123", "embedding": [0.1, 0.2, 0.3]},
            }
        ],
    }
    result = detect_geojson_type(geojson)
    assert result == "vector_layer"


def test_detect_geojson_type_label_without_embedding():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"id": "123", "label": 1},
            }
        ],
    }
    result = detect_geojson_type(geojson)
    assert result == "vector_layer"


def test_load_detections_preserves_existing_labels(sample_detection_geojson):
    state = AppState()
    state.pos_ids = ["existing_pos"]
    state.neg_ids = ["existing_neg"]
    state.cached_embeddings["existing_pos"] = np.array([1.0, 2.0, 3.0])

    dummy_data = DummyDataManager()
    dummy_map = DummyMapManager()
    manager = DatasetManager(dummy_data, dummy_map, state)

    manager._apply_detection_payload(sample_detection_geojson)

    assert "existing_pos" in state.pos_ids
    assert "existing_neg" in state.neg_ids
    assert "existing_pos" in state.cached_embeddings


def test_load_detections_enters_detection_mode():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TEST_TILE_1",
                    "probability": 0.75,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    assert state.detection_mode is False
    state.enter_detection_mode(detection_geojson)

    assert state.detection_mode is True
    assert state.detection_data == detection_geojson
    assert len(state.detection_labels) == 0


def test_label_detection_stores_in_detection_labels():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TEST_TILE_1",
                    "probability": 0.85,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    state.enter_detection_mode(detection_geojson)
    state.label_detection("TEST_TILE_1", UIConstants.POSITIVE_LABEL)

    assert "TEST_TILE_1" in state.detection_labels
    assert state.detection_labels["TEST_TILE_1"] == UIConstants.POSITIVE_LABEL


def test_label_detection_updates_existing_label():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TEST_TILE_1",
                    "probability": 0.9,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    state.enter_detection_mode(detection_geojson)
    state.label_detection("TEST_TILE_1", UIConstants.POSITIVE_LABEL)
    assert state.detection_labels["TEST_TILE_1"] == UIConstants.POSITIVE_LABEL

    state.label_detection("TEST_TILE_1", UIConstants.NEGATIVE_LABEL)
    assert state.detection_labels["TEST_TILE_1"] == UIConstants.NEGATIVE_LABEL


def test_exit_detection_mode_clears_state():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TEST_TILE_1",
                    "probability": 0.6,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    state.enter_detection_mode(detection_geojson)
    state.label_detection("TEST_TILE_1", UIConstants.POSITIVE_LABEL)
    assert state.detection_mode is True
    assert state.detection_data is not None
    assert len(state.detection_labels) == 1

    state.exit_detection_mode()

    assert state.detection_mode is False
    assert state.detection_data is None
    assert len(state.detection_labels) == 0


def test_get_labeled_detections():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TILE_A",
                    "probability": 0.7,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    state.enter_detection_mode(detection_geojson)
    state.label_detection("TILE_A", UIConstants.POSITIVE_LABEL)
    state.label_detection("TILE_B", UIConstants.NEGATIVE_LABEL)
    state.label_detection("TILE_C", UIConstants.POSITIVE_LABEL)

    labeled = state.get_labeled_detections()

    assert len(labeled) == 3
    assert ("TILE_A", UIConstants.POSITIVE_LABEL) in labeled
    assert ("TILE_B", UIConstants.NEGATIVE_LABEL) in labeled
    assert ("TILE_C", UIConstants.POSITIVE_LABEL) in labeled


def test_reset_clears_detection_mode():
    state = AppState()
    detection_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "tile_id": "TEST_TILE_1",
                    "probability": 0.8,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }

    state.enter_detection_mode(detection_geojson)
    state.label_detection("TEST_TILE_1", UIConstants.POSITIVE_LABEL)

    assert state.detection_mode is True
    assert state.detection_data is not None
    assert len(state.detection_labels) > 0

    state.reset()

    assert state.detection_mode is False
    assert state.detection_data is None
    assert len(state.detection_labels) == 0
