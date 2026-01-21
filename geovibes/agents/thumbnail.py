"""Satellite thumbnail generation for verification agent."""

import base64
import math
from io import BytesIO
from pathlib import Path

from PIL import Image

from geovibes.agents.schemas import ClusterInfo
from geovibes.ui.xyz import get_map_image

DEFAULT_BASEMAP = "GOOGLE_HYBRID"
DEFAULT_THUMBNAIL_SIZE_PX = 512
DEFAULT_BUFFER_M = 100


def generate_cluster_thumbnail(
    cluster: ClusterInfo,
    basemap: str = DEFAULT_BASEMAP,
    buffer_m: float = DEFAULT_BUFFER_M,
    output_size_px: int = DEFAULT_THUMBNAIL_SIZE_PX,
) -> bytes:
    """
    Generate a satellite thumbnail covering the cluster bounds.

    Parameters
    ----------
    cluster : ClusterInfo
        Cluster to generate thumbnail for
    basemap : str
        Basemap source (e.g., GOOGLE_HYBRID, MAPTILER)
    buffer_m : float
        Buffer in meters to add around cluster bounds
    output_size_px : int
        Output image size in pixels

    Returns
    -------
    bytes
        PNG image bytes
    """
    lat_range = cluster.bounds_max_lat - cluster.bounds_min_lat
    lon_range = cluster.bounds_max_lon - cluster.bounds_min_lon

    lat_m = lat_range * 111320
    lon_m = lon_range * 111320 * math.cos(math.radians(cluster.centroid_lat))
    coverage_m = max(lat_m, lon_m) + 2 * buffer_m

    coverage_m = max(coverage_m, 200)

    tile_spec = {
        "tile_size_px": output_size_px,
        "meters_per_pixel": coverage_m / output_size_px,
    }

    image_bytes = get_map_image(
        source=basemap,
        lon=cluster.centroid_lon,
        lat=cluster.centroid_lat,
        tile_spec=tile_spec,
    )

    return image_bytes


def generate_location_thumbnail(
    lat: float,
    lon: float,
    coverage_m: float = 500.0,
    basemap: str = DEFAULT_BASEMAP,
    output_size_px: int = DEFAULT_THUMBNAIL_SIZE_PX,
) -> bytes:
    """
    Generate a satellite thumbnail for a single location.

    Parameters
    ----------
    lat : float
        Latitude of the center point
    lon : float
        Longitude of the center point
    coverage_m : float
        Coverage area in meters
    basemap : str
        Basemap source
    output_size_px : int
        Output image size in pixels

    Returns
    -------
    bytes
        PNG image bytes
    """
    tile_spec = {
        "tile_size_px": output_size_px,
        "meters_per_pixel": coverage_m / output_size_px,
    }

    return get_map_image(
        source=basemap,
        lon=lon,
        lat=lat,
        tile_spec=tile_spec,
    )


def thumbnail_to_base64(image_bytes: bytes) -> str:
    """
    Convert image bytes to base64 string for LLM input.

    Parameters
    ----------
    image_bytes : bytes
        PNG image bytes

    Returns
    -------
    str
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def load_reference_image(path: str) -> str:
    """
    Load a reference image from disk and return as base64.

    Parameters
    ----------
    path : str
        Path to the reference image file

    Returns
    -------
    str
        Base64-encoded string
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    return base64.b64encode(image_bytes).decode("utf-8")


def resize_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Resize image if larger than max_size to reduce token usage.

    Parameters
    ----------
    image_bytes : bytes
        Original image bytes
    max_size : int
        Maximum dimension in pixels

    Returns
    -------
    bytes
        Resized image bytes (PNG format)
    """
    img = Image.open(BytesIO(image_bytes))

    if max(img.size) <= max_size:
        return image_bytes

    ratio = max_size / max(img.size)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def save_thumbnail(image_bytes: bytes, output_path: str) -> None:
    """
    Save thumbnail to disk for debugging.

    Parameters
    ----------
    image_bytes : bytes
        PNG image bytes
    output_path : str
        Path to save the image
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)
