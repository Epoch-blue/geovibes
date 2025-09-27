import pytest

from geovibes.ui import xyz


def test_deg2num_returns_expected_tile():
    xtile, ytile = xyz.deg2num(0.0, 0.0, 1)
    assert (xtile, ytile) == (1, 1)


def test_get_map_image_rejects_invalid_source():
    with pytest.raises(ValueError):
        xyz.get_map_image("INVALID", lon=0, lat=0)


def test_get_map_image_builds_url(monkeypatch):
    calls = {}

    def fake_get(url):
        calls["url"] = url

        class Response:
            content = b"payload"

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr(xyz.requests, "get", fake_get)

    lat = 10.0
    lon = 20.0
    zoom = 5
    xtile, ytile = xyz.deg2num(lat, lon, zoom)

    content = xyz.get_map_image("GOOGLE_HYBRID", lon=lon, lat=lat, zoom=zoom)

    assert content == b"payload"
    assert f"{zoom}" in calls["url"]
    assert f"x={xtile}" in calls["url"]
    assert f"y={ytile}" in calls["url"]


def test_compute_zoom_for_tile_matches_expected():
    coverage_m = 25 * 10
    zoom = xyz.compute_zoom_for_tile(0.0, coverage_m)
    assert zoom == 17


def test_get_map_image_uses_tile_spec_for_zoom(monkeypatch):
    calls = {}

    def fake_get(url):
        calls["url"] = url

        class Response:
            content = b"payload"

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr(xyz.requests, "get", fake_get)

    lat = 0.0
    lon = 0.0
    tile_spec = {"tile_size_px": 25, "tile_overlap_px": 0, "meters_per_pixel": 10}
    expected_zoom = xyz.compute_zoom_for_tile(lat, tile_spec["tile_size_px"] * tile_spec["meters_per_pixel"])

    xyz.get_map_image("GOOGLE_HYBRID", lon=lon, lat=lat, tile_spec=tile_spec)

    assert f"&z={expected_zoom}" in calls["url"]
