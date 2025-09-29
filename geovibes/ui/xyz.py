"""Map tile fetching utilities."""

import math
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import requests
import tenacity
from PIL import Image

from geovibes.ui_config import BasemapConfig

EARTH_RADIUS_M = 6_378_137
TILE_SIZE_PX = 256


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
    return max(0, min(22, int(round(zoom))))


def _xyz_sources() -> Dict[str, str]:
    """Return XYZ tile source templates that support x, y, z substitution."""

    return {
        key: value
        for key, value in BasemapConfig.BASEMAP_TILES.items()
        if "{z}" in value and "{x}" in value and "{y}" in value
    }


def _meters_per_pixel(lat_deg: float, zoom: int) -> Optional[float]:
    """Compute the ground resolution for a given latitude and zoom level."""

    lat_rad = math.radians(lat_deg)
    cos_lat = math.cos(lat_rad)
    if cos_lat <= 0:
        return None
    return 156543.03392 * cos_lat / (2 ** zoom)


def _tile_float_indices(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    """Return fractional XYZ tile indices for a geographic coordinate."""

    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x_float = (lon_deg + 180.0) / 360.0 * n
    y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x_float, y_float


@lru_cache(maxsize=512)
def _fetch_tile_bytes(source: str, template: str, zoom: int, x: int, y: int) -> bytes:
    """Download a single XYZ tile and cache the raw bytes."""

    url = template.format(z=zoom, x=x, y=y)
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def _assemble_centered_image(
    source: str,
    template: str,
    lat_deg: float,
    lon_deg: float,
    zoom: int,
    coverage_m: float,
) -> bytes:
    """Create a centered tile by stitching neighbors and cropping to coverage."""

    x_float, y_float = _tile_float_indices(lat_deg, lon_deg, zoom)
    base_x = int(math.floor(x_float))
    base_y = int(math.floor(y_float))
    frac_x = x_float - base_x
    frac_y = y_float - base_y
    n = 2 ** zoom
    x_offsets = [-1, 0, 1]
    y_candidates = [-1, 0, 1]
    x_tiles = [(offset, (base_x + offset) % n) for offset in x_offsets]
    y_tiles = []
    for offset in y_candidates:
        candidate = base_y + offset
        if 0 <= candidate < n:
            y_tiles.append((offset, candidate))
    if not y_tiles:
        y_tiles.append((0, min(max(base_y, 0), n - 1)))
    tile_width = None
    tile_height = None
    tile_images: Dict[Tuple[int, int], Image.Image] = {}
    for y_offset, y_index in y_tiles:
        for x_offset, x_index in x_tiles:
            tile_bytes = _fetch_tile_bytes(source, template, zoom, x_index, y_index)
            image = Image.open(BytesIO(tile_bytes))
            if tile_width is None or tile_height is None:
                tile_width, tile_height = image.size
            image = image.convert("RGB")
            tile_images[(x_offset, y_offset)] = image
    if tile_width is None or tile_height is None:
        tile_width = TILE_SIZE_PX
        tile_height = TILE_SIZE_PX

    width = len(x_tiles) * tile_width
    height = len(y_tiles) * tile_height
    mosaic = Image.new("RGB", (width, height))
    for row, (y_offset, y_index) in enumerate(y_tiles):
        for col, (x_offset, x_index) in enumerate(x_tiles):
            tile_image = tile_images[(x_offset, y_offset)]
            mosaic.paste(tile_image, (col * tile_width, row * tile_height))
    base_col = next(index for index, (offset, _) in enumerate(x_tiles) if offset == 0)
    base_row = next(index for index, (offset, _) in enumerate(y_tiles) if offset == 0)
    center_x = (base_col + frac_x) * tile_width
    center_y = (base_row + frac_y) * tile_height
    meters_per_pixel = _meters_per_pixel(lat_deg, zoom)
    if meters_per_pixel and tile_width:
        scale = TILE_SIZE_PX / float(tile_width)
        meters_per_pixel *= scale
    target_px = TILE_SIZE_PX
    if meters_per_pixel and coverage_m > 0:
        pixels_float = coverage_m / meters_per_pixel
        max_dimension = min(width, height)
        target_px = max(1, min(int(round(pixels_float)), max_dimension))
    half = target_px / 2.0
    left = center_x - half
    top = center_y - half
    left = max(0.0, min(left, width - target_px))
    top = max(0.0, min(top, height - target_px))
    left_int = int(round(left))
    top_int = int(round(top))
    right_int = left_int + target_px
    bottom_int = top_int + target_px
    if right_int > width:
        shift = right_int - width
        left_int = max(0, left_int - shift)
        right_int = left_int + target_px
    if bottom_int > height:
        shift = bottom_int - height
        top_int = max(0, top_int - shift)
        bottom_int = top_int + target_px
    cropped = mosaic.crop((left_int, top_int, right_int, bottom_int))
    if cropped.size != (TILE_SIZE_PX, TILE_SIZE_PX):
        cropped = cropped.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.Resampling.BILINEAR)
    if cropped.size != (TILE_SIZE_PX, TILE_SIZE_PX):
        cropped = cropped.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.Resampling.BILINEAR)
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
)
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
    valid_sources = _xyz_sources()

    if source not in valid_sources:
        raise ValueError(
            f"Invalid XYZ tile source: {source}. Valid sources are: {list(valid_sources.keys())}"
        )

    template = valid_sources[source]

    if zoom is None and tile_spec:
        size_px = tile_spec.get("tile_size_px")
        resolution = tile_spec.get("meters_per_pixel")
        if size_px and resolution:
            coverage = size_px * resolution
            inferred_zoom = compute_zoom_for_tile(lat, coverage)
            if inferred_zoom is not None:
                zoom = inferred_zoom

    if zoom is None:
        zoom = 17

    if tile_spec:
        size_px = tile_spec.get("tile_size_px")
        resolution = tile_spec.get("meters_per_pixel")
        if size_px and resolution:
            coverage = size_px * resolution
            return _assemble_centered_image(
                source=source,
                template=template,
                lat_deg=lat,
                lon_deg=lon,
                zoom=zoom,
                coverage_m=coverage,
            )

    xtile, ytile = deg2num(lat, lon, zoom)
    tile_bytes = _fetch_tile_bytes(source, template, zoom, xtile, ytile)
    return tile_bytes
