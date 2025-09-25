import numpy as np

from geovibes.ui.state import AppState
from geovibes.ui_config import UIConstants


def test_apply_label_toggle():
    state = AppState()
    result = state.apply_label("1", UIConstants.POSITIVE_LABEL)
    assert result == "positive"
    assert "1" in state.pos_ids

    result = state.apply_label("1", UIConstants.POSITIVE_LABEL)
    assert result == "removed"
    assert "1" not in state.pos_ids


def test_update_query_vector_uses_cached_embeddings():
    state = AppState()
    state.pos_ids.append("pt")
    state.cached_embeddings["pt"] = np.array([1.0, 2.0, 3.0])
    vector = state.update_query_vector()
    assert vector is not None
    np.testing.assert_array_equal(vector, np.array([2.0, 4.0, 6.0]))


def test_reset_restores_defaults():
    state = AppState()
    state.pos_ids.extend(["1", "2"])
    state.neg_ids.append("3")
    state.tile_basemap = "S2_RGB"
    state.selection_mode = "polygon"
    state.query_vector = np.array([1, 2, 3])

    state.reset()

    assert state.pos_ids == []
    assert state.neg_ids == []
    assert state.query_vector is None
    assert state.tile_basemap == "MAPTILER"
    assert state.selection_mode == "point"
    assert state.current_label == "Positive"
