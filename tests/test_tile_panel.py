import time
from types import SimpleNamespace

import pandas as pd
from ipywidgets import Button, Label, VBox

from geovibes.ui.app import GeoVibes
from geovibes.ui.state import AppState
from geovibes.ui.tiles import TilePanel


class DummyMapManager:
    def __init__(self, tile_spec=None):
        self.controls = []
        self.operations = []
        self.data = SimpleNamespace(tile_spec=tile_spec)

    def add_widget_control(self, widget, position="topright"):
        control = SimpleNamespace(widget=widget, position=position)
        self.controls.append(control)
        return control

    def set_operation(self, message):
        self.operations.append(message)

    def clear_operation(self):
        self.operations.append(None)


def _await_tiles(panel: TilePanel, timeout: float = 1.0) -> None:
    end = time.time() + timeout
    while (panel._pending_batches or panel._pending_futures) and time.time() < end:
        time.sleep(0.01)


def test_tile_panel_update_results(monkeypatch):
    state = AppState()
    map_manager = DummyMapManager()
    panel = TilePanel(
        state=state,
        map_manager=map_manager,
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    captured_ids = []

    def fake_widget(self, row, image_bytes, rank=None):
        captured_ids.append(row["id"])
        return VBox([Label(value=f"tile-{row['id']}")])

    monkeypatch.setattr(TilePanel, "_build_tile_widget", fake_widget)
    monkeypatch.setattr(TilePanel, "_fetch_tile_image_bytes", lambda self, row: None)

    df = pd.DataFrame(
        [
            {"id": "1"},
            {"id": "2"},
        ]
    )

    ready_calls = []

    panel.update_results(df, auto_show=True, on_ready=lambda: ready_calls.append(True))

    _await_tiles(panel)

    assert panel.container.layout.display == "flex"
    assert captured_ids == ["1", "2"]
    assert state.last_search_results_df is df
    assert state.tile_page == 0
    assert panel.next_tiles_btn.layout.display != "flex"
    assert map_manager.operations[:2] == ["⏳ Loading tiles...", "✅ Tiles updated"]
    assert panel.results_ready is True
    assert ready_calls == [True]

    panel.toggle()
    assert panel.container.layout.display == "none"


def test_tile_panel_handle_map_basemap_change(monkeypatch):
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    calls = []

    def fake_render(
        self, append, limit=None, on_finish=None, page_size=None, loader_token=None
    ):
        calls.append((append, limit, on_finish, page_size, loader_token))
        if on_finish:
            on_finish()

    monkeypatch.setattr(TilePanel, "_render_current_page", fake_render)

    state.last_search_results_df = pd.DataFrame([{"id": "1"}])
    panel.results_grid.children = (Label(value="existing"),)

    panel.handle_map_basemap_change(panel.allowed_sources[0])

    _await_tiles(panel)

    assert state.tile_basemap == panel.allowed_sources[0]
    assert calls == [(False, 1, None, None, None)]


def test_tile_panel_apply_label_style_updates_buttons():
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    state.pos_ids.append("1")
    state.neg_ids.append("2")

    tick = Button()
    cross = Button()

    panel._apply_label_style("1", tick, cross)

    assert tick.button_style == "success"
    assert cross.button_style == ""

    panel._apply_label_style("2", tick, cross)

    assert tick.button_style == ""
    assert cross.button_style == "danger"


def test_next_tiles_loads_more_results(monkeypatch):
    state = AppState()
    state.initial_load_size = 4
    state.tiles_per_page = 4
    map_manager = DummyMapManager()

    panel = TilePanel(
        state=state,
        map_manager=map_manager,
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    def fake_widget(self, row, image_bytes, rank=None):
        return Label(value=f"tile-{row['id']}")

    monkeypatch.setattr(TilePanel, "_build_tile_widget", fake_widget)
    monkeypatch.setattr(TilePanel, "_fetch_tile_image_bytes", lambda self, row: None)

    df = pd.DataFrame([{"id": str(i)} for i in range(8)])

    panel.update_results(df, auto_show=True)
    _await_tiles(panel)

    assert panel._page_sizes == [4]
    assert len(panel.results_grid.children) == 4

    panel._on_next_tiles_click(None)

    _await_tiles(panel)

    assert panel._page_sizes == [4, 4]
    assert len(panel.results_grid.children) == 8
    assert map_manager.operations[-2:] == ["⏳ Loading tiles...", "✅ Tiles updated"]


def test_tile_panel_passes_tile_spec_to_get_map_image(monkeypatch):
    state = AppState()
    tile_spec = {
        "tile_size_px": 25,
        "tile_overlap_px": 0,
        "meters_per_pixel": 10,
    }
    map_manager = DummyMapManager(tile_spec=tile_spec)

    panel = TilePanel(
        state=state,
        map_manager=map_manager,
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    captured = {}

    def fake_get_map_image(source, lon, lat, zoom=None, tile_spec=None):
        captured["tile_spec"] = tile_spec
        captured["zoom"] = zoom
        return b"bytes"

    monkeypatch.setattr("geovibes.ui.tiles.get_map_image", fake_get_map_image)

    df = pd.DataFrame(
        [
            {
                "id": "1",
                "geometry_wkt": "POINT(0 0)",
            }
        ]
    )

    panel.update_results(df, auto_show=True)

    _await_tiles(panel)

    assert captured["tile_spec"] == tile_spec
    assert captured["zoom"] is None


def test_handle_tile_center_uses_tile_spec(monkeypatch):
    tile_spec = {"tile_size_px": 25, "meters_per_pixel": 10}

    class DummyMapManager:
        def __init__(self):
            self.data = SimpleNamespace(tile_spec=tile_spec)
            self.center_calls = []
            self.highlight_calls = []

        def center_on(self, lat, lon, zoom=None):
            self.center_calls.append((lat, lon, zoom))

        def highlight_polygon(self, polygon, *, color="red", fill_opacity=0.0):
            self.highlight_calls.append((polygon, color, fill_opacity))

    app = GeoVibes.__new__(GeoVibes)
    app.data = SimpleNamespace(tile_spec=tile_spec)
    app.map_manager = DummyMapManager()
    app.state = AppState()
    app._show_operation_status = lambda *args, **kwargs: None

    row = {"geometry_wkt": "POINT(0 0)"}

    app._handle_tile_center(row)

    assert app.map_manager.center_calls

    polygon, color, fill_opacity = app.map_manager.highlight_calls[-1]
    assert color == "red"
    assert fill_opacity == 0.0

    half_side_deg = (tile_spec["tile_size_px"] * tile_spec["meters_per_pixel"]) / (
        2 * 111_320
    )
    minx, miny, maxx, maxy = polygon.bounds
    tolerance = 1e-5
    assert abs(minx + half_side_deg) < tolerance
    assert abs(maxx - half_side_deg) < tolerance
    assert abs(miny + half_side_deg) < tolerance
    assert abs(maxy - half_side_deg) < tolerance


def test_sort_toggle_initial_state():
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    assert panel._sort_order == "Similar"
    assert "sort-btn-active" in panel.similar_btn._dom_classes
    assert "sort-btn-inactive" in panel.dissimilar_btn._dom_classes


def test_sort_toggle_switches_to_dissimilar(monkeypatch):
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    render_calls = []

    def fake_render(
        self, append, limit=None, on_finish=None, page_size=None, loader_token=None
    ):
        render_calls.append(("render", self._sort_order))
        if on_finish:
            on_finish()

    monkeypatch.setattr(TilePanel, "_render_current_page", fake_render)

    state.last_search_results_df = pd.DataFrame([{"id": "1"}])

    panel._on_dissimilar_click(None)

    assert panel._sort_order == "Dissimilar"
    assert "sort-btn-active" in panel.dissimilar_btn._dom_classes
    assert "sort-btn-inactive" in panel.similar_btn._dom_classes
    assert len(render_calls) == 1


def test_sort_toggle_switches_back_to_similar(monkeypatch):
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    render_calls = []

    def fake_render(
        self, append, limit=None, on_finish=None, page_size=None, loader_token=None
    ):
        render_calls.append(("render", self._sort_order))
        if on_finish:
            on_finish()

    monkeypatch.setattr(TilePanel, "_render_current_page", fake_render)

    state.last_search_results_df = pd.DataFrame([{"id": "1"}])

    panel._on_dissimilar_click(None)
    panel._on_similar_click(None)

    assert panel._sort_order == "Similar"
    assert "sort-btn-active" in panel.similar_btn._dom_classes
    assert "sort-btn-inactive" in panel.dissimilar_btn._dom_classes
    assert len(render_calls) == 2


def test_sort_toggle_no_op_when_already_selected(monkeypatch):
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    render_calls = []

    def fake_render(
        self, append, limit=None, on_finish=None, page_size=None, loader_token=None
    ):
        render_calls.append("render")
        if on_finish:
            on_finish()

    monkeypatch.setattr(TilePanel, "_render_current_page", fake_render)

    state.last_search_results_df = pd.DataFrame([{"id": "1"}])

    panel._on_similar_click(None)

    assert panel._sort_order == "Similar"
    assert len(render_calls) == 0


def _extract_tile_ids(panel: TilePanel) -> list:
    """Extract tile IDs from the results grid children in display order."""
    ids = []
    for child in panel.results_grid.children:
        if isinstance(child, VBox) and child.children:
            label = child.children[0]
            if isinstance(label, Label) and label.value.startswith("tile-"):
                ids.append(label.value.replace("tile-", ""))
    return ids


def test_dissimilar_reverses_tile_order(monkeypatch):
    state = AppState()
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    def fake_widget(self, row, image_bytes, rank=None):
        return VBox([Label(value=f"tile-{row['id']}")])

    monkeypatch.setattr(TilePanel, "_build_tile_widget", fake_widget)
    monkeypatch.setattr(TilePanel, "_fetch_tile_image_bytes", lambda self, row: None)

    df = pd.DataFrame([{"id": "1"}, {"id": "2"}, {"id": "3"}])

    panel._sort_order = "Similar"
    panel.update_results(df, auto_show=True)
    _await_tiles(panel)
    similar_order = _extract_tile_ids(panel)

    panel._on_dissimilar_click(None)
    _await_tiles(panel)
    dissimilar_order = _extract_tile_ids(panel)

    assert similar_order == ["1", "2", "3"]
    assert dissimilar_order == ["3", "2", "1"]


def test_detection_mode_inverts_sort_order(monkeypatch):
    """In detection mode, data is sorted by probability ascending (low=dissimilar).

    Similar should reverse to show high prob first.
    Dissimilar should keep order to show low prob first.
    """
    state = AppState()
    state.detection_mode = True
    panel = TilePanel(
        state=state,
        map_manager=DummyMapManager(),
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    def fake_widget(self, row, image_bytes, rank=None):
        return VBox([Label(value=f"tile-{row['id']}")])

    monkeypatch.setattr(TilePanel, "_build_tile_widget", fake_widget)
    monkeypatch.setattr(TilePanel, "_fetch_tile_image_bytes", lambda self, row: None)

    df = pd.DataFrame([{"id": "1"}, {"id": "2"}, {"id": "3"}])

    panel._sort_order = "Similar"
    panel.update_results(df, auto_show=True)
    _await_tiles(panel)
    similar_order = _extract_tile_ids(panel)

    panel._on_dissimilar_click(None)
    _await_tiles(panel)
    dissimilar_order = _extract_tile_ids(panel)

    assert similar_order == ["3", "2", "1"]
    assert dissimilar_order == ["1", "2", "3"]
