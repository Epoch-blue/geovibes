from types import SimpleNamespace

import pytest

from geovibes.ui.map_manager import MapManager


def test_update_basemap_updates_url():
    manager = MapManager.__new__(MapManager)
    manager.basemap_tiles = {"A": "url-a", "B": "url-b"}
    manager.basemap_layer = SimpleNamespace(url="url-a")
    manager.current_basemap = "A"

    manager.update_basemap("B")

    assert manager.basemap_layer.url == "url-b"
    assert manager.current_basemap == "B"


def test_update_basemap_rejects_unknown():
    manager = MapManager.__new__(MapManager)
    manager.basemap_tiles = {"A": "url-a"}
    manager.basemap_layer = SimpleNamespace(url="url-a")
    manager.current_basemap = "A"

    with pytest.raises(ValueError):
        manager.update_basemap("missing")


def test_update_status_uses_status_bus():
    class DummyStatusBus:
        def __init__(self):
            self.calls = []

        def render(self, **kwargs):
            self.calls.append(kwargs)
            return "rendered"

        def set_operation(self, message):
            self.last_operation = message

        def clear_operation(self):
            self.last_operation = None

    manager = MapManager.__new__(MapManager)
    manager.status_bus = DummyStatusBus()
    manager.map = SimpleNamespace(center=(5.0, 6.0))
    manager.state = SimpleNamespace(
        lasso_mode=True,
        current_label="Positive",
        polygon_drawing=False,
    )
    manager.status_bar = SimpleNamespace(value="")

    manager.update_status()

    assert manager.status_bar.value == "rendered"
    assert manager.status_bus.calls[0]["mode"] == "Polygon"
    assert manager.status_bus.calls[0]["lat"] == 5.0
    assert manager.status_bus.calls[0]["lon"] == 6.0

    manager.set_operation("Working")
    assert manager.status_bus.last_operation == "Working"
    assert manager.status_bar.value == "rendered"

    manager.clear_operation()
    assert manager.status_bus.last_operation is None
    assert manager.status_bar.value == "rendered"
