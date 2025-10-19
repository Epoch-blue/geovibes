from geovibes.ui.status import StatusBus


def test_status_bus_render_includes_operation():
    bus = StatusBus()
    bus.set_operation("Loading")

    html = bus.render(lat=1.23456, lon=2.34567, mode="Point", label="Positive")

    assert "Loading" in html
    assert "Lat: 1.2346" in html
    assert "Lon: 2.3457" in html

    bus.clear_operation()
    html = bus.render(
        lat=0, lon=0, mode="Polygon", label="Negative", polygon_drawing=True
    )

    assert "Drawing polygon" in html
