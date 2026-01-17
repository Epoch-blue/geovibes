"""Export CDL aquaculture/water mask to GeoTIFF via Earth Engine.

Takes detection polygons, buffers them, intersects with CDL water classes,
applies morphological closing, and exports to Google Drive.
"""

import argparse
import json
import time

import ee
import geopandas as gpd


def load_geojson_to_ee(geojson_path: str, buffer_m: float) -> ee.Geometry:
    """Load GeoJSON polygons, buffer, and convert to EE geometry."""
    start = time.perf_counter()

    gdf = gpd.read_file(geojson_path)
    print(f"  Loaded {len(gdf)} features from {geojson_path}")

    # Buffer in a projected CRS (UTM zone estimated from centroid)
    centroid = gdf.union_all().centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_crs = (
        f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
    )

    gdf_utm = gdf.to_crs(utm_crs)
    gdf_buffered = gdf_utm.buffer(buffer_m)
    gdf_buffered_wgs84 = gpd.GeoSeries(gdf_buffered, crs=utm_crs).to_crs("EPSG:4326")

    # Union all geometries
    union_geom = gdf_buffered_wgs84.union_all()

    # Convert to EE geometry
    geojson_dict = json.loads(gpd.GeoSeries([union_geom]).to_json())
    ee_geom = ee.Geometry(geojson_dict["features"][0]["geometry"])

    elapsed = time.perf_counter() - start
    print(f"  Buffered by {buffer_m}m and converted to EE geometry ({elapsed:.2f}s)")

    return ee_geom


def create_cdl_water_mask(year: int) -> ee.Image:
    """Create binary mask for CDL aquaculture (92) and open water (111)."""
    start = time.perf_counter()

    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("cropland")
    )

    # Binary mask: Aquaculture (92) OR Open Water (111)
    water_mask = cdl.eq(92).Or(cdl.eq(111))

    elapsed = time.perf_counter() - start
    print(f"  Created CDL water mask for {year} ({elapsed:.2f}s)")

    return water_mask


def apply_morphological_closing(mask: ee.Image, radius: int = 1) -> ee.Image:
    """Apply morphological closing (dilation then erosion)."""
    start = time.perf_counter()

    closed = mask.focal_max(
        radius=radius, kernelType="square", units="pixels"
    ).focal_min(radius=radius, kernelType="square", units="pixels")

    elapsed = time.perf_counter() - start
    print(
        f"  Applied morphological closing with {2*radius+1}x{2*radius+1} kernel ({elapsed:.2f}s)"
    )

    return closed


def export_to_drive(
    image: ee.Image,
    region: ee.Geometry,
    output_name: str,
    scale: int = 30,
    folder: str = "earth_engine_exports",
) -> ee.batch.Task:
    """Export image to Google Drive as GeoTIFF."""
    start = time.perf_counter()

    # Cast to byte (0/1 values)
    image_byte = image.toByte()

    task = ee.batch.Export.image.toDrive(
        image=image_byte,
        description=output_name,
        folder=folder,
        fileNamePrefix=output_name,
        region=region,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    task.start()

    elapsed = time.perf_counter() - start
    print(f"  Started export task '{output_name}' ({elapsed:.2f}s)")

    return task


def main():
    parser = argparse.ArgumentParser(
        description="Export CDL aquaculture/water mask to GeoTIFF"
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to detection GeoJSON (polygons)",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name for the export (used in Drive filename)",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=160,
        help="Buffer distance in meters (default: 160)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="CDL year to use (default: 2023)",
    )
    parser.add_argument(
        "--folder",
        default="earth_engine_exports",
        help="Google Drive folder for export (default: earth_engine_exports)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print info without starting export",
    )

    args = parser.parse_args()

    print("Initializing Earth Engine...")
    ee.Initialize()

    print("\n1. Loading and buffering detections...")
    region = load_geojson_to_ee(args.geojson, args.buffer_m)

    print("\n2. Creating CDL water mask...")
    water_mask = create_cdl_water_mask(args.year)

    print("\n3. Applying morphological closing...")
    closed_mask = apply_morphological_closing(water_mask, radius=1)

    # Clip to region
    clipped_mask = closed_mask.clip(region)

    if args.dry_run:
        print("\n[DRY RUN] Would export to Drive:")
        print(f"  Output name: {args.output_name}")
        print(f"  Folder: {args.folder}")
        print("  Scale: 30m")
        return

    print("\n4. Exporting to Google Drive...")
    task = export_to_drive(
        clipped_mask,
        region,
        args.output_name,
        scale=30,
        folder=args.folder,
    )

    print("\n✅ Export task started!")
    print(f"   Task ID: {task.id}")
    print(f"   Status: {task.status()['state']}")
    print("   Monitor at: https://code.earthengine.google.com/tasks")
    print(
        f"\n   Output will appear in Google Drive folder: {args.folder}/{args.output_name}.tif"
    )


if __name__ == "__main__":
    main()
