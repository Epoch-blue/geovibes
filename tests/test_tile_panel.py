from types import SimpleNamespace

import pandas as pd
from ipywidgets import Button, Label, VBox

from geovibes.ui.state import AppState
from geovibes.ui.tiles import TilePanel


class DummyMapManager:
    def __init__(self):
        self.controls = []
        self.operations = []

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
