from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geovibes.ui.map_manager import MapManager


# ------------------------------------------------------------------
# Overlay layer tests
# ------------------------------------------------------------------


def _setup_layer_manager_mocks(manager):
    """Set up mock layer manager widget attributes for testing."""
    manager._layer_rows = SimpleNamespace(children=[])
    container = SimpleNamespace(
        _visible=False,
        show=lambda: setattr(container, "_visible", True),
        hide=lambda: setattr(container, "_visible", False),
    )
    manager._layer_manager_container = container
    manager._create_layer_row = lambda name, opacity: SimpleNamespace(name=name)


def test_add_tile_layer_creates_layer():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.basemap_layer = SimpleNamespace(name="basemap")
    manager.map = SimpleNamespace(layers=(manager.basemap_layer,))
    _setup_layer_manager_mocks(manager)

    def mock_insert(layer):
        manager.map.layers = manager.map.layers + (layer,)

    manager._insert_overlay_layer = mock_insert

    manager.add_tile_layer("http://tiles/{z}/{x}/{y}.png", "test_layer", 0.8)

    assert "test_layer" in manager._overlay_layers
    assert manager._overlay_layers["test_layer"].opacity == 0.8
    assert manager._overlay_layers["test_layer"].url == "http://tiles/{z}/{x}/{y}.png"


def test_add_tile_layer_duplicate_name_raises():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {"existing": SimpleNamespace()}

    with pytest.raises(ValueError, match="already exists"):
        manager.add_tile_layer("http://tiles/{z}/{x}/{y}.png", "existing")


def test_add_tile_layer_clamps_opacity():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.basemap_layer = SimpleNamespace(name="basemap")
    manager.map = SimpleNamespace(layers=(manager.basemap_layer,))
    _setup_layer_manager_mocks(manager)
    manager._insert_overlay_layer = lambda layer: None

    manager.add_tile_layer("http://tiles/{z}/{x}/{y}.png", "high", 1.5)
    assert manager._overlay_layers["high"].opacity == 1.0

    manager.add_tile_layer("http://tiles/{z}/{x}/{y}.png", "low", -0.5)
    assert manager._overlay_layers["low"].opacity == 0.0


def test_remove_layer_returns_false_for_unknown():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}

    result = manager.remove_layer("nonexistent")

    assert result is False


def test_remove_layer_removes_from_map():
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(name="test")
    manager._overlay_layers = {"test": layer}
    _setup_layer_manager_mocks(manager)

    removed_layers = []
    manager.map = SimpleNamespace(
        layers=(layer,),
        remove_layer=lambda lyr: removed_layers.append(lyr),
    )

    result = manager.remove_layer("test")

    assert result is True
    assert "test" not in manager._overlay_layers
    assert layer in removed_layers


def test_set_layer_opacity():
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(opacity=1.0)
    manager._overlay_layers = {"test": layer}

    manager.set_layer_opacity("test", 0.5)

    assert layer.opacity == 0.5


def test_set_layer_opacity_clamps_values():
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(opacity=1.0)
    manager._overlay_layers = {"test": layer}

    manager.set_layer_opacity("test", 1.5)
    assert layer.opacity == 1.0

    manager.set_layer_opacity("test", -0.5)
    assert layer.opacity == 0.0


def test_set_layer_opacity_unknown_raises():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}

    with pytest.raises(ValueError, match="not found"):
        manager.set_layer_opacity("nonexistent", 0.5)


def test_list_overlay_layers():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {
        "layer1": SimpleNamespace(),
        "layer2": SimpleNamespace(),
    }

    result = manager.list_overlay_layers()

    assert set(result) == {"layer1", "layer2"}


def test_clear_overlay_layers():
    manager = MapManager.__new__(MapManager)
    layer1 = SimpleNamespace(name="layer1")
    layer2 = SimpleNamespace(name="layer2")
    manager._overlay_layers = {"layer1": layer1, "layer2": layer2}
    _setup_layer_manager_mocks(manager)

    removed = []
    manager.map = SimpleNamespace(
        layers=(layer1, layer2),
        remove_layer=lambda lyr: removed.append(lyr),
    )

    manager.clear_overlay_layers()

    assert manager._overlay_layers == {}
    assert len(removed) == 2


def test_insert_overlay_layer_positions_after_basemap():
    manager = MapManager.__new__(MapManager)
    basemap = SimpleNamespace(name="basemap")
    geojson = SimpleNamespace(name="points")
    manager.basemap_layer = basemap
    manager.map = SimpleNamespace(layers=(basemap, geojson))

    new_layer = SimpleNamespace(name="overlay")
    # Layer is added to _overlay_layers before _insert_overlay_layer is called
    manager._overlay_layers = {"overlay": new_layer}
    manager._insert_overlay_layer(new_layer)

    layers = list(manager.map.layers)
    assert layers.index(new_layer) == 1
    assert layers.index(basemap) == 0
    assert layers.index(geojson) == 2


def test_insert_multiple_overlay_layers_maintains_order():
    """New overlay layers should appear on top of existing overlays."""
    manager = MapManager.__new__(MapManager)
    basemap = SimpleNamespace(name="basemap")
    geojson = SimpleNamespace(name="points")
    manager.basemap_layer = basemap
    manager.map = SimpleNamespace(layers=(basemap, geojson))
    manager._overlay_layers = {}

    layer1 = SimpleNamespace(name="layer1")
    manager._overlay_layers["layer1"] = layer1
    manager._insert_overlay_layer(layer1)

    layer2 = SimpleNamespace(name="layer2")
    manager._overlay_layers["layer2"] = layer2
    manager._insert_overlay_layer(layer2)

    layer3 = SimpleNamespace(name="layer3")
    manager._overlay_layers["layer3"] = layer3
    manager._insert_overlay_layer(layer3)

    layers = list(manager.map.layers)
    assert layers == [basemap, layer1, layer2, layer3, geojson]
    assert layers.index(basemap) == 0
    assert layers.index(layer1) == 1
    assert layers.index(layer2) == 2
    assert layers.index(layer3) == 3
    assert layers.index(geojson) == 4


def test_add_ee_layer_raises_when_ee_unavailable():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.data = SimpleNamespace(ee_available=False)

    with pytest.raises(RuntimeError, match="Earth Engine not available"):
        manager.add_ee_layer(MagicMock(), {"min": 0, "max": 1}, "ee_layer")


def test_add_ee_layer_success(monkeypatch):
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.data = SimpleNamespace(ee_available=True)
    manager.basemap_layer = SimpleNamespace(name="basemap")
    manager.map = SimpleNamespace(layers=(manager.basemap_layer,))
    _setup_layer_manager_mocks(manager)
    manager._insert_overlay_layer = lambda layer: None

    monkeypatch.setattr(
        "geovibes.ui.map_manager.get_ee_image_url",
        lambda img, params: "https://earthengine.googleapis.com/tile",
    )

    manager.add_ee_layer(MagicMock(), {"min": 0, "max": 1}, "ndvi", 0.6)

    assert "ndvi" in manager._overlay_layers
    assert manager._overlay_layers["ndvi"].opacity == 0.6
    assert "earthengine" in manager._overlay_layers["ndvi"].url


def test_refresh_layer_manager_shows_widget_when_layers_exist():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {"test": SimpleNamespace(opacity=0.5)}
    _setup_layer_manager_mocks(manager)

    manager._refresh_layer_manager()

    assert manager._layer_manager_container._visible is True
    assert len(manager._layer_rows.children) == 1


def test_refresh_layer_manager_hides_widget_when_no_layers():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    _setup_layer_manager_mocks(manager)
    manager._layer_manager_container._visible = True

    manager._refresh_layer_manager()

    assert manager._layer_manager_container._visible is False
    assert len(manager._layer_rows.children) == 0


def test_add_tile_layer_refreshes_layer_manager():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.basemap_layer = SimpleNamespace(name="basemap")
    manager.map = SimpleNamespace(layers=(manager.basemap_layer,))
    _setup_layer_manager_mocks(manager)

    refresh_calls = []

    def mock_refresh():
        refresh_calls.append(True)
        manager._layer_manager_container._visible = True

    manager._refresh_layer_manager = mock_refresh
    manager._insert_overlay_layer = lambda layer: None

    manager.add_tile_layer("http://tiles/{z}/{x}/{y}.png", "test", 0.7)

    assert len(refresh_calls) == 1


def test_remove_layer_refreshes_layer_manager():
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(name="test")
    manager._overlay_layers = {"test": layer}
    manager.map = SimpleNamespace(
        layers=(layer,),
        remove_layer=lambda lyr: None,
    )
    _setup_layer_manager_mocks(manager)
    manager._layer_manager_container._visible = True

    refresh_calls = []

    def mock_refresh():
        refresh_calls.append(True)
        manager._layer_manager_container._visible = False

    manager._refresh_layer_manager = mock_refresh

    manager.remove_layer("test")

    assert len(refresh_calls) == 1


def test_remove_layer_not_in_map_layers():
    """Test remove_layer when layer exists in dict but not in map.layers."""
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(name="orphan")
    manager._overlay_layers = {"orphan": layer}
    _setup_layer_manager_mocks(manager)

    remove_called = []
    manager.map = SimpleNamespace(
        layers=(),
        remove_layer=lambda lyr: remove_called.append(lyr),
    )

    result = manager.remove_layer("orphan")

    assert result is True
    assert "orphan" not in manager._overlay_layers
    assert len(remove_called) == 0


def test_clear_overlay_layers_refreshes_layer_manager():
    manager = MapManager.__new__(MapManager)
    layer = SimpleNamespace(name="test")
    manager._overlay_layers = {"test": layer}
    manager.map = SimpleNamespace(
        layers=(layer,),
        remove_layer=lambda lyr: None,
    )
    _setup_layer_manager_mocks(manager)
    manager._layer_manager_container._visible = True

    refresh_calls = []

    def mock_refresh():
        refresh_calls.append(True)
        manager._layer_manager_container._visible = False

    manager._refresh_layer_manager = mock_refresh

    manager.clear_overlay_layers()

    assert len(refresh_calls) == 1
    assert manager._layer_manager_container._visible is False


def test_add_tile_layer_with_attribution():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}
    manager.basemap_layer = SimpleNamespace(name="basemap")
    manager.map = SimpleNamespace(layers=(manager.basemap_layer,))
    _setup_layer_manager_mocks(manager)
    manager._insert_overlay_layer = lambda layer: None

    manager.add_tile_layer(
        "http://tiles/{z}/{x}/{y}.png",
        "attributed",
        opacity=1.0,
        attribution="© OpenStreetMap",
    )

    assert manager._overlay_layers["attributed"].attribution == "© OpenStreetMap"


def test_create_layer_row_truncates_long_names():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}

    row = manager._create_layer_row("VeryLongLayerName", 0.5)

    label = row.children[0]
    assert label.children[0] == "VeryLongL"


def test_create_layer_row_preserves_short_names():
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}

    row = manager._create_layer_row("LULC2024", 0.8)

    label = row.children[0]
    assert label.children[0] == "LULC2024"


def test_ipyvuetify_show_hide_methods():
    """Verify ipyvuetify widgets have working show/hide methods."""
    import ipyvuetify as v

    card = v.Card()
    assert hasattr(card, "show")
    assert hasattr(card, "hide")

    card.hide()
    assert "d-none" in card.class_

    card.show()
    assert "d-none" not in (card.class_ or "")


def test_layer_manager_visibility_toggle():
    """Test that layer manager container visibility is properly toggled."""
    manager = MapManager.__new__(MapManager)
    manager._overlay_layers = {}

    manager._layer_rows = MagicMock()
    manager._layer_rows.children = []

    import ipyvuetify as v

    manager._layer_manager_container = v.Card()
    manager._create_layer_row = lambda name, opacity: MagicMock()

    manager._layer_manager_container.hide()
    assert "d-none" in manager._layer_manager_container.class_

    manager._overlay_layers = {"test": MagicMock(opacity=0.5)}
    manager._refresh_layer_manager()
    assert "d-none" not in (manager._layer_manager_container.class_ or "")

    manager._overlay_layers = {}
    manager._refresh_layer_manager()
    assert "d-none" in manager._layer_manager_container.class_


# ------------------------------------------------------------------
# Basemap tests
# ------------------------------------------------------------------


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
