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


def test_tile_panel_update_results(monkeypatch):
    state = AppState()
    map_manager = DummyMapManager()
    panel = TilePanel(
        state=state,
        map_manager=map_manager,
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    def fake_widget(self, row, append):
        return Label(value=f"tile-{row['id']}")

    monkeypatch.setattr(TilePanel, "_create_tile_widget", fake_widget)

    df = pd.DataFrame([
        {"id": "1"},
        {"id": "2"},
    ])

    panel.update_results(df)

    assert panel.container.layout.display == "block"
    assert [child.value for child in panel.results_grid.children] == [
        "tile-1",
        "tile-2",
    ]
    assert state.last_search_results_df is df
    assert state.tile_page == 0
    assert panel.next_tiles_btn.layout.display != "flex"
    assert map_manager.operations[:2] == ["⏳ Loading tiles...", "✅ Tiles updated"]

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

    def fake_render(self, append, limit=None):
        calls.append((append, limit))

    monkeypatch.setattr(TilePanel, "_render_current_page", fake_render)

    state.last_search_results_df = pd.DataFrame([{"id": "1"}])
    panel.results_grid.children = (Label(value="existing"),)

    panel.handle_map_basemap_change(panel.allowed_sources[0])

    assert state.tile_basemap == panel.allowed_sources[0]
    assert calls == [(False, 1)]


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


def test_next_tiles_shows_loading_placeholders(monkeypatch):
    state = AppState()
    state.initial_load_size = 1
    state.tiles_per_page = 1
    map_manager = DummyMapManager()

    panel = TilePanel(
        state=state,
        map_manager=map_manager,
        on_label=lambda *args, **kwargs: None,
        on_center=lambda row: None,
    )

    observed = {"placeholder_seen": False}

    def fake_widget(self, row, append):
        if any(
            isinstance(child, VBox)
            and any(
                isinstance(grandchild, Label) and grandchild.value == "Loading..."
                for grandchild in child.children
            )
            for child in self.results_grid.children
        ):
            observed["placeholder_seen"] = True
        return Label(value=f"tile-{row['id']}")

    monkeypatch.setattr(TilePanel, "_create_tile_widget", fake_widget)

    df = pd.DataFrame([
        {"id": "1"},
        {"id": "2"},
    ])

    panel.update_results(df)
    observed["placeholder_seen"] = False

    panel._on_next_tiles_click(None)

    assert observed["placeholder_seen"] is True
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

    panel.update_results(df)

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

    half_side_deg = (tile_spec["tile_size_px"] * tile_spec["meters_per_pixel"]) / (2 * 111_320)
    minx, miny, maxx, maxy = polygon.bounds
    tolerance = 1e-5
    assert abs(minx + half_side_deg) < tolerance
    assert abs(maxx - half_side_deg) < tolerance
    assert abs(miny + half_side_deg) < tolerance
    assert abs(maxy - half_side_deg) < tolerance
