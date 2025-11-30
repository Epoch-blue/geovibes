#!/usr/bin/env python3
"""
Benchmark different tile fetching strategies.

Tests:
1. Current implementation (sequential sub-tile fetching)
2. Parallel sub-tile fetching with ThreadPoolExecutor
3. HTTP Session with connection pooling
4. aiohttp async fetching
5. httpx with HTTP/2
6. Combined optimizations

Usage:
    uv run python benchmark_tile_fetching.py
"""

import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

# Optional async libraries - gracefully handle if not installed
try:
    import aiohttp
    import asyncio

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Constants
EARTH_RADIUS_M = 6_378_137
TILE_SIZE_PX = 256

# Test configuration
TEST_POINTS = [
    # Alabama - various locations
    (32.3792, -86.3077),  # Montgomery
    (33.5207, -86.8025),  # Birmingham
    (30.6954, -88.0399),  # Mobile
    (34.7304, -86.5861),  # Huntsville
    (32.4610, -86.4106),  # Near Montgomery
]

# MapTiler API key from environment
MAPTILER_API_KEY = os.environ.get("MAPTILER_API_KEY", "")
TILE_TEMPLATE = f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}"

# Tile spec for 320m coverage (32px @ 10m resolution)
TILE_SPEC = {"tile_size_px": 32, "meters_per_pixel": 10}
COVERAGE_M = 320  # 32 * 10


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert degrees to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def compute_zoom_for_tile(lat_deg: float, coverage_m: float) -> int:
    """Compute optimal zoom level for given coverage."""
    lat_rad = math.radians(lat_deg)
    cos_lat = math.cos(lat_rad)
    numerator = cos_lat * 2 * math.pi * EARTH_RADIUS_M
    zoom = math.log2(numerator / coverage_m)
    return max(0, min(22, int(round(zoom))))


def _tile_float_indices(
    lat_deg: float, lon_deg: float, zoom: int
) -> Tuple[float, float]:
    """Return fractional XYZ tile indices for a geographic coordinate."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    x_float = (lon_deg + 180.0) / 360.0 * n
    y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x_float, y_float


def _meters_per_pixel(lat_deg: float, zoom: int) -> float:
    """Compute the ground resolution for a given latitude and zoom level."""
    lat_rad = math.radians(lat_deg)
    cos_lat = math.cos(lat_rad)
    return 156543.03392 * cos_lat / (2**zoom)


def get_tile_coords(
    lat: float, lon: float, zoom: int
) -> List[Tuple[int, int, int, int]]:
    """Get list of (x, y, x_offset, y_offset) for 3x3 grid around point."""
    x_float, y_float = _tile_float_indices(lat, lon, zoom)
    base_x = int(math.floor(x_float))
    base_y = int(math.floor(y_float))
    n = 2**zoom

    tiles = []
    for y_offset in [-1, 0, 1]:
        y_idx = base_y + y_offset
        if 0 <= y_idx < n:
            for x_offset in [-1, 0, 1]:
                x_idx = (base_x + x_offset) % n
                tiles.append((x_idx, y_idx, x_offset, y_offset))
    return tiles


# =============================================================================
# Method 1: Current implementation (sequential, no session)
# =============================================================================


def fetch_tile_sequential(zoom: int, x: int, y: int) -> bytes:
    """Fetch a single tile using requests (no session)."""
    url = TILE_TEMPLATE.format(z=zoom, x=x, y=y)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def assemble_image_sequential(lat: float, lon: float, zoom: int) -> bytes:
    """Current implementation - sequential tile fetching."""
    tiles = get_tile_coords(lat, lon, zoom)
    tile_images = {}

    for x, y, x_off, y_off in tiles:
        tile_bytes = fetch_tile_sequential(zoom, x, y)
        img = Image.open(BytesIO(tile_bytes)).convert("RGB")
        tile_images[(x_off, y_off)] = img

    return _assemble_mosaic(lat, lon, zoom, tile_images)


# =============================================================================
# Method 2: Parallel sub-tile fetching with ThreadPoolExecutor
# =============================================================================


def assemble_image_parallel_threads(lat: float, lon: float, zoom: int) -> bytes:
    """Parallel tile fetching using ThreadPoolExecutor."""
    tiles = get_tile_coords(lat, lon, zoom)
    tile_images = {}

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {
            executor.submit(fetch_tile_sequential, zoom, x, y): (x_off, y_off)
            for x, y, x_off, y_off in tiles
        }
        for future in as_completed(futures):
            x_off, y_off = futures[future]
            tile_bytes = future.result()
            img = Image.open(BytesIO(tile_bytes)).convert("RGB")
            tile_images[(x_off, y_off)] = img

    return _assemble_mosaic(lat, lon, zoom, tile_images)


# =============================================================================
# Method 3: HTTP Session with connection pooling
# =============================================================================

# Global session for connection reuse
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def fetch_tile_session(zoom: int, x: int, y: int) -> bytes:
    """Fetch a single tile using session with connection pooling."""
    session = get_session()
    url = TILE_TEMPLATE.format(z=zoom, x=x, y=y)
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def assemble_image_session_sequential(lat: float, lon: float, zoom: int) -> bytes:
    """Session-based sequential fetching."""
    tiles = get_tile_coords(lat, lon, zoom)
    tile_images = {}

    for x, y, x_off, y_off in tiles:
        tile_bytes = fetch_tile_session(zoom, x, y)
        img = Image.open(BytesIO(tile_bytes)).convert("RGB")
        tile_images[(x_off, y_off)] = img

    return _assemble_mosaic(lat, lon, zoom, tile_images)


def assemble_image_session_parallel(lat: float, lon: float, zoom: int) -> bytes:
    """Session-based parallel fetching."""
    tiles = get_tile_coords(lat, lon, zoom)
    tile_images = {}

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {
            executor.submit(fetch_tile_session, zoom, x, y): (x_off, y_off)
            for x, y, x_off, y_off in tiles
        }
        for future in as_completed(futures):
            x_off, y_off = futures[future]
            tile_bytes = future.result()
            img = Image.open(BytesIO(tile_bytes)).convert("RGB")
            tile_images[(x_off, y_off)] = img

    return _assemble_mosaic(lat, lon, zoom, tile_images)


# =============================================================================
# Method 4: aiohttp async fetching
# =============================================================================

if HAS_AIOHTTP:

    async def fetch_tile_aiohttp(
        session: aiohttp.ClientSession, zoom: int, x: int, y: int
    ) -> Tuple[int, int, bytes]:
        """Fetch a single tile using aiohttp."""
        url = TILE_TEMPLATE.format(z=zoom, x=x, y=y)
        async with session.get(url) as response:
            response.raise_for_status()
            return (x, y, await response.read())

    async def assemble_image_aiohttp_async(lat: float, lon: float, zoom: int) -> bytes:
        """Async tile fetching using aiohttp."""
        tiles = get_tile_coords(lat, lon, zoom)

        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_tile_aiohttp(session, zoom, x, y) for x, y, _, _ in tiles]
            results = await asyncio.gather(*tasks)

        # Map results back to offsets
        tile_images = {}
        for (x, y, x_off, y_off), (_, _, tile_bytes) in zip(tiles, results):
            img = Image.open(BytesIO(tile_bytes)).convert("RGB")
            tile_images[(x_off, y_off)] = img

        return _assemble_mosaic(lat, lon, zoom, tile_images)

    def assemble_image_aiohttp(lat: float, lon: float, zoom: int) -> bytes:
        """Wrapper to run async aiohttp in sync context."""
        return asyncio.run(assemble_image_aiohttp_async(lat, lon, zoom))


# =============================================================================
# Method 5: httpx with HTTP/2
# =============================================================================

if HAS_HTTPX:

    def assemble_image_httpx_sync(lat: float, lon: float, zoom: int) -> bytes:
        """Sync httpx with connection pooling."""
        tiles = get_tile_coords(lat, lon, zoom)
        tile_images = {}

        with httpx.Client(http2=True, timeout=10.0) as client:
            for x, y, x_off, y_off in tiles:
                url = TILE_TEMPLATE.format(z=zoom, x=x, y=y)
                response = client.get(url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                tile_images[(x_off, y_off)] = img

        return _assemble_mosaic(lat, lon, zoom, tile_images)

    async def assemble_image_httpx_async_impl(
        lat: float, lon: float, zoom: int
    ) -> bytes:
        """Async httpx with HTTP/2."""
        tiles = get_tile_coords(lat, lon, zoom)

        async with httpx.AsyncClient(http2=True, timeout=10.0) as client:
            tasks = []
            for x, y, x_off, y_off in tiles:
                url = TILE_TEMPLATE.format(z=zoom, x=x, y=y)
                tasks.append(client.get(url))
            responses = await asyncio.gather(*tasks)

        tile_images = {}
        for (x, y, x_off, y_off), response in zip(tiles, responses):
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            tile_images[(x_off, y_off)] = img

        return _assemble_mosaic(lat, lon, zoom, tile_images)

    def assemble_image_httpx_async(lat: float, lon: float, zoom: int) -> bytes:
        """Wrapper to run async httpx in sync context."""
        return asyncio.run(assemble_image_httpx_async_impl(lat, lon, zoom))


# =============================================================================
# Common mosaic assembly
# =============================================================================


def _assemble_mosaic(
    lat: float, lon: float, zoom: int, tile_images: Dict[Tuple[int, int], Image.Image]
) -> bytes:
    """Assemble tiles into final centered image."""
    tile_width = TILE_SIZE_PX
    tile_height = TILE_SIZE_PX

    # Get unique offsets
    x_offsets = sorted(set(k[0] for k in tile_images.keys()))
    y_offsets = sorted(set(k[1] for k in tile_images.keys()))

    width = len(x_offsets) * tile_width
    height = len(y_offsets) * tile_height
    mosaic = Image.new("RGB", (width, height))

    for (x_off, y_off), img in tile_images.items():
        col = x_offsets.index(x_off)
        row = y_offsets.index(y_off)
        mosaic.paste(img, (col * tile_width, row * tile_height))

    # Calculate center and crop
    x_float, y_float = _tile_float_indices(lat, lon, zoom)
    base_x = int(math.floor(x_float))
    base_y = int(math.floor(y_float))
    frac_x = x_float - base_x
    frac_y = y_float - base_y

    base_col = x_offsets.index(0) if 0 in x_offsets else 0
    base_row = y_offsets.index(0) if 0 in y_offsets else 0

    center_x = (base_col + frac_x) * tile_width
    center_y = (base_row + frac_y) * tile_height

    meters_per_px = _meters_per_pixel(lat, zoom)
    target_px = max(1, min(int(round(COVERAGE_M / meters_per_px)), min(width, height)))

    half = target_px / 2.0
    left = max(0, int(center_x - half))
    top = max(0, int(center_y - half))
    right = min(width, left + target_px)
    bottom = min(height, top + target_px)

    cropped = mosaic.crop((left, top, right, bottom))
    if cropped.size != (TILE_SIZE_PX, TILE_SIZE_PX):
        cropped = cropped.resize(
            (TILE_SIZE_PX, TILE_SIZE_PX), Image.Resampling.BILINEAR
        )

    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


# =============================================================================
# Benchmark runner
# =============================================================================


def benchmark_method(
    name: str, func, points: List[Tuple[float, float]], iterations: int = 3
):
    """Benchmark a tile fetching method."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    all_times = []
    for iteration in range(iterations):
        iteration_times = []
        for lat, lon in points:
            zoom = compute_zoom_for_tile(lat, COVERAGE_M)
            start = time.perf_counter()
            try:
                func(lat, lon, zoom)  # Execute and discard result
                elapsed = time.perf_counter() - start
                iteration_times.append(elapsed)
            except Exception as e:
                print(f"  Error at ({lat}, {lon}): {e}")
                iteration_times.append(float("inf"))

        avg = statistics.mean([t for t in iteration_times if t != float("inf")])
        print(f"  Iteration {iteration + 1}: avg={avg*1000:.1f}ms per tile")
        all_times.extend(iteration_times)

    valid_times = [t for t in all_times if t != float("inf")]
    if valid_times:
        mean_time = statistics.mean(valid_times)
        std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
        min_time = min(valid_times)
        max_time = max(valid_times)
        print(f"\n  Summary ({len(valid_times)} samples):")
        print(f"    Mean:   {mean_time*1000:.1f}ms")
        print(f"    Std:    {std_time*1000:.1f}ms")
        print(f"    Min:    {min_time*1000:.1f}ms")
        print(f"    Max:    {max_time*1000:.1f}ms")
        return mean_time
    return float("inf")


def benchmark_batch_fetching(
    name: str, func, points: List[Tuple[float, float]], batch_size: int = 5
):
    """Benchmark fetching multiple tiles in parallel (simulating TilePanel behavior)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Batch ({batch_size} concurrent): {name}")
    print(f"{'='*60}")

    zoom = compute_zoom_for_tile(points[0][0], COVERAGE_M)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(func, lat, lon, zoom) for lat, lon in points]
        for f in as_completed(futures):
            f.result()  # Wait for completion, discard result
    elapsed = time.perf_counter() - start

    print(f"  Total time for {len(points)} tiles: {elapsed*1000:.1f}ms")
    print(f"  Average per tile: {elapsed/len(points)*1000:.1f}ms")
    return elapsed


def main():
    if not MAPTILER_API_KEY:
        print("ERROR: MAPTILER_API_KEY environment variable not set")
        print("Set it with: export MAPTILER_API_KEY=your_key")
        return

    print("Tile Fetching Benchmark")
    print("=" * 60)
    print(f"Test points: {len(TEST_POINTS)}")
    print(f"Coverage: {COVERAGE_M}m")
    print(f"Tile template: {TILE_TEMPLATE[:60]}...")

    # Warm up DNS and connections
    print("\nWarming up...")
    zoom = compute_zoom_for_tile(TEST_POINTS[0][0], COVERAGE_M)
    try:
        assemble_image_sequential(TEST_POINTS[0][0], TEST_POINTS[0][1], zoom)
    except Exception as e:
        print(f"Warmup failed: {e}")

    results = {}

    # Method 1: Sequential (current implementation)
    results["1_sequential"] = benchmark_method(
        "1. Sequential (current)", assemble_image_sequential, TEST_POINTS
    )

    # Method 2: Parallel threads
    results["2_parallel_threads"] = benchmark_method(
        "2. Parallel ThreadPool", assemble_image_parallel_threads, TEST_POINTS
    )

    # Method 3a: Session sequential
    results["3a_session_seq"] = benchmark_method(
        "3a. Session + Sequential", assemble_image_session_sequential, TEST_POINTS
    )

    # Method 3b: Session parallel
    results["3b_session_parallel"] = benchmark_method(
        "3b. Session + Parallel", assemble_image_session_parallel, TEST_POINTS
    )

    # Method 4: aiohttp
    if HAS_AIOHTTP:
        results["4_aiohttp"] = benchmark_method(
            "4. aiohttp async", assemble_image_aiohttp, TEST_POINTS
        )
    else:
        print("\n[SKIP] aiohttp not installed")

    # Method 5a: httpx sync - skip due to h2 dependency issues
    # if HAS_HTTPX:
    #     results["5a_httpx_sync"] = benchmark_method(
    #         "5a. httpx sync (HTTP/2)", assemble_image_httpx_sync, TEST_POINTS
    #     )
    #     results["5b_httpx_async"] = benchmark_method(
    #         "5b. httpx async (HTTP/2)", assemble_image_httpx_async, TEST_POINTS
    #     )
    # else:
    print("\n[SKIP] httpx HTTP/2 - h2 package issues")

    # Batch benchmark (simulating TilePanel with 8 workers)
    print("\n" + "=" * 60)
    print("BATCH BENCHMARKS (simulating TilePanel with concurrent tiles)")
    print("=" * 60)

    batch_results = {}
    batch_results["batch_sequential"] = benchmark_batch_fetching(
        "Sequential inner", assemble_image_sequential, TEST_POINTS, batch_size=8
    )
    batch_results["batch_parallel"] = benchmark_batch_fetching(
        "Parallel inner", assemble_image_parallel_threads, TEST_POINTS, batch_size=8
    )
    batch_results["batch_session_parallel"] = benchmark_batch_fetching(
        "Session + Parallel inner",
        assemble_image_session_parallel,
        TEST_POINTS,
        batch_size=8,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Single Tile Assembly Time (ms)")
    print("=" * 60)
    baseline = results.get("1_sequential", 1)
    for name, time_sec in sorted(results.items()):
        speedup = baseline / time_sec if time_sec > 0 else 0
        print(f"  {name:25s}: {time_sec*1000:7.1f}ms  ({speedup:.2f}x)")

    print("\n" + "=" * 60)
    print("SUMMARY - Batch Fetching Time (5 tiles)")
    print("=" * 60)
    baseline_batch = batch_results.get("batch_sequential", 1)
    for name, time_sec in sorted(batch_results.items()):
        speedup = baseline_batch / time_sec if time_sec > 0 else 0
        print(f"  {name:25s}: {time_sec*1000:7.1f}ms  ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
