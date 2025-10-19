from io import BytesIO

import pytest
from PIL import Image

from geovibes.ui import xyz


def test_deg2num_returns_expected_tile():
    xtile, ytile = xyz.deg2num(0.0, 0.0, 1)
    assert (xtile, ytile) == (1, 1)


def test_get_map_image_rejects_invalid_source():
    with pytest.raises(ValueError):
        xyz.get_map_image("INVALID", lon=0, lat=0)


def test_get_map_image_builds_url(monkeypatch):
    captured = {}

    def fake_fetch(source, template, zoom, x, y):
        captured["args"] = (source, template, zoom, x, y)
        return b"payload"

    monkeypatch.setattr(xyz, "_fetch_tile_bytes", fake_fetch)

    lat = 10.0
    lon = 20.0
    zoom = 5
    xtile, ytile = xyz.deg2num(lat, lon, zoom)

    content = xyz.get_map_image("GOOGLE_HYBRID", lon=lon, lat=lat, zoom=zoom)

    assert content == b"payload"
    assert captured["args"] == (
        "GOOGLE_HYBRID",
        xyz._xyz_sources()["GOOGLE_HYBRID"],
        zoom,
        xtile,
        ytile,
    )


def test_compute_zoom_for_tile_matches_expected():
    coverage_m = 25 * 10
    zoom = xyz.compute_zoom_for_tile(0.0, coverage_m)
    assert zoom == 17


def test_get_map_image_uses_tile_spec_for_zoom(monkeypatch):
    captured = {}

    def fake_assemble(source, template, lat_deg, lon_deg, zoom, coverage_m):
        captured["args"] = (source, template, lat_deg, lon_deg, zoom, coverage_m)
        return b"assembled"

    monkeypatch.setattr(xyz, "_assemble_centered_image", fake_assemble)

    lat = 0.0
    lon = 0.0
    tile_spec = {"tile_size_px": 25, "tile_overlap_px": 0, "meters_per_pixel": 10}
    expected_coverage = tile_spec["tile_size_px"] * tile_spec["meters_per_pixel"]
    expected_zoom = xyz.compute_zoom_for_tile(lat, expected_coverage)

    content = xyz.get_map_image("GOOGLE_HYBRID", lon=lon, lat=lat, tile_spec=tile_spec)

    template = xyz._xyz_sources()["GOOGLE_HYBRID"]
    assert content == b"assembled"
    assert captured["args"] == (
        "GOOGLE_HYBRID",
        template,
        lat,
        lon,
        expected_zoom,
        expected_coverage,
    )


def test_assemble_centered_image_reads_neighbors(monkeypatch):
    zoom = 5
    lat = 0.0
    lon = 20.0
    template = xyz._xyz_sources()["GOOGLE_HYBRID"]
    base_x, base_y = xyz.deg2num(lat, lon, zoom)

    def tile_bytes(source, template_arg, zoom_arg, x_arg, y_arg):
        assert zoom_arg == zoom
        assert template_arg == template
        color = (x_arg % 256, y_arg % 256, (x_arg + y_arg) % 256)
        buffer = BytesIO()
        Image.new("RGB", (256, 256), color).save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(xyz, "_fetch_tile_bytes", tile_bytes)

    coverage = 1000.0
    image_bytes = xyz._assemble_centered_image(
        source="GOOGLE_HYBRID",
        template=template,
        lat_deg=lat,
        lon_deg=lon,
        zoom=zoom,
        coverage_m=coverage,
    )

    image = Image.open(BytesIO(image_bytes))
    assert image.size == (256, 256)
    center_color = image.getpixel((128, 128))
    west_color = image.getpixel((0, 128))
    east_color = image.getpixel((255, 128))
    north_color = image.getpixel((128, 0))
    south_color = image.getpixel((128, 255))

    n = 2**zoom
    neighbor_west = ((base_x - 1) % n, base_y)
    neighbor_east = ((base_x + 1) % n, base_y)
    neighbor_north = (base_x, max(base_y - 1, 0))
    neighbor_south = (base_x, min(base_y + 1, n - 1))

    def expected_color(x_arg: int, y_arg: int) -> tuple[int, int, int]:
        return (x_arg % 256, y_arg % 256, (x_arg + y_arg) % 256)

    def matches_expected(color: tuple[int, int, int], coord: tuple[int, int]) -> bool:
        expected = expected_color(*coord)
        return all(abs(a - b) <= 1 for a, b in zip(color, expected))

    assert matches_expected(center_color, (base_x, base_y))
    assert matches_expected(west_color, neighbor_west)
    assert matches_expected(east_color, neighbor_east)
    assert matches_expected(north_color, neighbor_north)
    assert matches_expected(south_color, neighbor_south)
