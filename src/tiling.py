"""
Generate a grid of tiles over an MGRS tile and save it as a GeoParquet file.
"""
import argparse
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pyproj
import shapely.geometry
import shapely.ops
import pandas as pd
import pyproj
import os
import tempfile
import zipfile
import subprocess
from google.cloud import storage
from tqdm import tqdm
import shutil


def get_crs_from_tile(tile_series: pd.Series) -> str:
    """
    Get the CRS from a tile series by reading the 'epsg' column.
    """
    try:
        epsg_code = tile_series['epsg']
        return f"EPSG:{epsg_code}"
    except KeyError:
        raise ValueError("Input series must have an 'epsg' column.")


@dataclass
class MGRSTileGrid:
    """Class for tracking a MGRS tile grid"""
    mgrs_tile_id: str
    crs: str
    tilesize: int
    overlap: int
    resolution: float
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = f"{self.mgrs_tile_id}_{self.crs.split(':')[-1]}_{self.tilesize}_{self.overlap}_{int(self.resolution)}"


def chip_mgrs_tile(tile_series: pd.Series, mgrs_tile_grid: MGRSTileGrid, source_crs: pyproj.CRS) -> gpd.GeoDataFrame:
    """
    Top level function to generate chips over an MGRS tile
    """
    xform_utm = pyproj.Transformer.from_crs(source_crs, mgrs_tile_grid.crs, always_xy=True)
    tile_geom_utm = shapely.ops.transform(xform_utm.transform, tile_series.geometry)

    eff_tilesize = mgrs_tile_grid.tilesize * mgrs_tile_grid.resolution
    eff_overlap = mgrs_tile_grid.overlap * mgrs_tile_grid.resolution
    grid_spacing = eff_tilesize - eff_overlap

    bounds_utm = tile_geom_utm.bounds
    sw_utm = bounds_utm[0], bounds_utm[1]
    ne_utm = bounds_utm[2], bounds_utm[3]

    x_diff = ne_utm[0] - sw_utm[0]
    y_diff = ne_utm[1] - sw_utm[1]

    x_samples = round(x_diff / grid_spacing) + 1
    y_samples = round(y_diff / grid_spacing) + 1

    xs = np.arange(0, x_samples) * grid_spacing + sw_utm[0]
    ys = np.arange(0, y_samples) * grid_spacing + sw_utm[1]

    x_grid, y_grid = np.meshgrid(xs, ys)

    return generate_chips(
        x_samples=x_samples,
        y_samples=y_samples,
        x_grid=x_grid,
        y_grid=y_grid,
        eff_tilesize=eff_tilesize,
        mgrs_tile_grid=mgrs_tile_grid,
        tile_geom_utm=tile_geom_utm,
    )


def generate_chips(
    x_samples: int,
    y_samples: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    eff_tilesize: float,
    mgrs_tile_grid: MGRSTileGrid,
    tile_geom_utm: shapely.geometry.Polygon,
) -> gpd.GeoDataFrame:
    """
    Generate chips over a grid and return them as a GeoDataFrame.
    """
    tiles = []
    for i in range(x_samples):
        for j in range(y_samples):
            x, y = x_grid[j, i], y_grid[j, i]
            geom = shapely.geometry.Point(x, y).buffer(eff_tilesize / 2, cap_style=3)

            if tile_geom_utm.intersects(geom):
                tile = {
                    'geometry': geom,
                    'tile_id': f"{mgrs_tile_grid.mgrs_tile_id}_{mgrs_tile_grid.tilesize}_{mgrs_tile_grid.overlap}_{int(mgrs_tile_grid.resolution)}_{j}_{i}"
                }
                tiles.append(tile)

    return gpd.GeoDataFrame(tiles, crs=mgrs_tile_grid.crs)


def write_tiles_to_geoparquet(tiles: gpd.GeoDataFrame, tile_name: str, output_dir: str = "."):
    """
    Write a GeoDataFrame of chips to a GeoParquet file locally.
    """
    output_path = f"{output_dir}/{tile_name}.parquet"
    tiles.to_parquet(output_path)
    print(f"Wrote {len(tiles)} tiles to {output_path}")


def upload_chips_to_gee(tiles: gpd.GeoDataFrame, tile_name: str, gcs_bucket: str, gee_asset_path: str, output_dir: str):
    """
    Writes chips to a zipped shapefile, uploads to GCS, and ingests into GEE.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save shapefile to temp dir
        shapefile_path = os.path.join(tmpdir, f"{tile_name}.shp")
        tiles.to_file(shapefile_path, driver='ESRI Shapefile')

        # Zip the shapefile components
        zip_path = os.path.join(tmpdir, f"{tile_name}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                source_file = f"{shapefile_path[:-4]}{ext}"
                if os.path.exists(source_file):
                    zipf.write(source_file, arcname=os.path.basename(source_file))

        # Save zipped shapefile locally
        output_zip_path = os.path.join(output_dir, os.path.basename(zip_path))
        shutil.copy(zip_path, output_zip_path)
        print(f"    Saved zipped shapefile to {output_zip_path}")

        # # Upload to GCS with progress bar
        # storage_client = storage.Client()
        # bucket = storage_client.bucket(gcs_bucket)
        # blob_name = f"{tile_name}.zip"
        # blob = bucket.blob(blob_name)
        #
        # file_size = os.path.getsize(zip_path)
        # with open(zip_path, "rb") as f, tqdm.wrapattr(
        #     f, "read", total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
        #     desc=f"  Uploading {os.path.basename(zip_path)}"
        # ) as file_obj:
        #     blob.upload_from_file(file_obj, size=file_size, content_type='application/zip')
        #
        # gcs_uri = f"gs://{gcs_bucket}/{blob_name}"
        # print(f"\n    Successfully uploaded to {gcs_uri}")
        #
        # # Upload to GEE
        # asset_id = f"{gee_asset_path}/{tile_name}"
        # print(f"    Uploading to GEE with asset ID: {asset_id}")
        # command = [
        #     'earthengine', 'upload', 'table',
        #     f'--asset_id={asset_id}',
        #     gcs_uri
        # ]
        # try:
        #     result = subprocess.run(command, check=True, capture_output=True, text=True)
        #     print(f"    GEE upload started successfully. Task ID: {result.stdout.strip()}")
        # except subprocess.CalledProcessError as e:
        #     print(f"    Error starting GEE upload: {e.stderr}")
        #     raise e


def main():
    parser = argparse.ArgumentParser(description="Generate tiling grid for MGRS tiles from a file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to GeoParquet or GeoJSON file with MGRS tile geometries.")
    parser.add_argument("--roi_file", type=str, help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles.")
    parser.add_argument("--tilesize", type=int, default=32, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels.")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters per pixel.")
    parser.add_argument("--buffer_m", type=float, default=500.0, help="Buffer distance in meters for post-filtering chips against the ROI.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output GeoParquet files.")
    parser.add_argument("--gcs_bucket", type=str, default='geovibes', help="GCS bucket to upload zipped shapefiles to.")
    parser.add_argument("--gee_asset_path", type=str, default='projects/demeterlabs-gee/assets/tiles', help="GEE asset path for table uploads (e.g., 'users/username/folder').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.input_file.endswith(".parquet"):
            mgrs_gdf = gpd.read_parquet(args.input_file)
        elif args.input_file.endswith(".geojson"):
            mgrs_gdf = gpd.read_file(args.input_file)
        else:
            raise ValueError("Input file must be a .parquet or .geojson file.")
    except Exception as e:
        raise IOError(f"Could not read input file: {args.input_file}") from e

    roi_geometry = None
    if args.roi_file:
        print(f"Filtering MGRS tiles by ROI: {args.roi_file}")
        try:
            if args.roi_file.endswith(".parquet"):
                roi_gdf = gpd.read_parquet(args.roi_file)
            elif args.roi_file.endswith(".geojson"):
                roi_gdf = gpd.read_file(args.roi_file)
            else:
                raise ValueError("ROI file must be a .parquet or .geojson file.")
        except Exception as e:
            raise IOError(f"Could not read ROI file: {args.roi_file}") from e

        if mgrs_gdf.crs != roi_gdf.crs:
            print(f"Warning: MGRS file CRS ({mgrs_gdf.crs}) and ROI file CRS ({roi_gdf.crs}) differ. Reprojecting ROI to match MGRS for intersection.")
            roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)

        roi_geometry = roi_gdf.union_all()
        intersecting_mask = mgrs_gdf.intersects(roi_geometry)
        mgrs_gdf = mgrs_gdf[intersecting_mask]
        print(f"Found {len(mgrs_gdf)} MGRS tiles intersecting with the ROI.")

    for _, tile_series in mgrs_gdf.iterrows():
        try:
            print(f"\nProcessing MGRS tile: {tile_series.mgrs_id}")
            crs = get_crs_from_tile(tile_series)
            grid = MGRSTileGrid(
                mgrs_tile_id=tile_series.mgrs_id,
                crs=crs,
                tilesize=args.tilesize,
                overlap=args.overlap,
                resolution=args.resolution,
            )
            tiles = chip_mgrs_tile(tile_series, grid, source_crs=mgrs_gdf.crs)
            print(f"    Generated {len(tiles)} initial chips.")

            if roi_geometry and len(tiles) > 0:
                transformer = pyproj.Transformer.from_crs(roi_gdf.crs, grid.crs, always_xy=True)
                roi_utm = shapely.ops.transform(transformer.transform, roi_geometry)
                buffered_roi_utm = roi_utm.buffer(args.buffer_m)
                
                initial_chip_count = len(tiles)
                intersecting_mask = tiles.intersects(buffered_roi_utm)
                tiles = tiles[intersecting_mask]
                print(f"    Post-filtering: Kept {len(tiles)} of {initial_chip_count} chips intersecting with the {args.buffer_m}m buffered ROI.")

            if len(tiles) > 0:
                if args.gcs_bucket and args.gee_asset_path:
                    upload_chips_to_gee(tiles, grid.prefix, args.gcs_bucket, args.gee_asset_path, args.output_dir)
                else:
                    write_tiles_to_geoparquet(tiles, grid.prefix, args.output_dir)
            else:
                print(f"    No chips to save for tile {tile_series.mgrs_id} after filtering.")

        except Exception as e:
            print(f"Could not process tile {tile_series.get('mgrs_id', 'N/A')}: {e}")


if __name__ == "__main__":
    main() 