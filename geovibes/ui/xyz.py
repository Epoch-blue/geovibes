"""Map tile fetching utilities."""

import math
from io import BytesIO
from typing import Optional, Dict, Any

import requests
import tenacity
from PIL import Image

from geovibes.ui_config import BasemapConfig

EARTH_RADIUS_M = 6_378_137


def deg2num(lat_deg: float, lon_deg: float, zoom: int):
    """Convert degrees to tile numbers.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        zoom: Zoom level.

    Returns:
        A tuple of (xtile, ytile).
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
)
def compute_zoom_for_tile(lat_deg: float, coverage_m: float) -> Optional[int]:
    if coverage_m <= 0:
        return None
    lat_rad = math.radians(lat_deg)
    cos_lat = math.cos(lat_rad)
    if cos_lat <= 0:
        return None
    numerator = cos_lat * 2 * math.pi * EARTH_RADIUS_M
    if numerator <= 0:
        return None
    zoom = math.log2(numerator / coverage_m)
    if not math.isfinite(zoom):
        return None
    zoom_int = int(math.floor(zoom))
    return max(0, min(22, zoom_int))


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
)
def _tile_coverage_m(lat_deg: float, zoom: int) -> float:
    lat_rad = math.radians(lat_deg)
    cos_lat = max(math.cos(lat_rad), 1e-6)
    return cos_lat * 2 * math.pi * EARTH_RADIUS_M / (2.0 ** zoom)


def get_map_image(
    source: str,
    lon: float,
    lat: float,
    zoom: Optional[int] = None,
    tile_spec: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Fetch a map tile image for a given coordinate and source.

    Args:
        source: The tile source (e.g., 'MAPTILER', 'GOOGLE_HYBRID').
        lon: Longitude of the point.
        lat: Latitude of the point.
        zoom: The zoom level for the tile.

    Returns:
        The image content as bytes.
    
    Raises:
        ValueError: If the source is not a valid XYZ tile source.
        requests.exceptions.RequestException: If the image download fails.
    """
    # Filter for XYZ tile sources, excluding static map APIs
    valid_sources = {
        k: v
        for k, v in BasemapConfig.BASEMAP_TILES.items()
        if "{z}" in v and "{x}" in v and "{y}" in v
    }

    if source not in valid_sources:
        raise ValueError(
            f"Invalid XYZ tile source: {source}. Valid sources are: {list(valid_sources.keys())}"
        )

    coverage_target = None
    if tile_spec:
        size_px = tile_spec.get("tile_size_px")
        resolution = tile_spec.get("meters_per_pixel")
        if size_px and resolution and size_px > 0 and resolution > 0:
            coverage_target = size_px * resolution

    if zoom is None and coverage_target:
        inferred_zoom = compute_zoom_for_tile(lat, coverage_target)
        if inferred_zoom is not None:
            zoom = inferred_zoom

    if zoom is None:
        zoom = 17
    else:
        zoom = max(0, min(22, zoom))

    if coverage_target:
        while zoom > 0 and _tile_coverage_m(lat, zoom) < coverage_target:
            zoom -= 1

    url_template = valid_sources[source]
    xtile, ytile = deg2num(lat, lon, zoom)
    url = url_template.format(z=zoom, x=xtile, y=ytile)

    response = requests.get(url)
    response.raise_for_status()
    image_bytes = response.content

    if not coverage_target:
        return image_bytes

    current_coverage = _tile_coverage_m(lat, zoom)
    if current_coverage <= 0:
        return image_bytes

    crop_ratio = coverage_target / current_coverage
    if crop_ratio >= 1 or crop_ratio <= 0:
        return image_bytes

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            crop_size = max(1, min(width, height, int(round(width * crop_ratio))))
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            box = (left, top, left + crop_size, top + crop_size)
            cropped = img.crop(box).resize((width, height), Image.BILINEAR)
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return image_bytes

    return image_bytes
