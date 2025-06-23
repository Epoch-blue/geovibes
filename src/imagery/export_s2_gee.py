#!/usr/bin/env python3

import argparse
import os
import sys
import json
from pathlib import Path

import ee
import geopandas as gpd
from shapely.geometry import shape

sys.path.append(str(Path(__file__).parent.parent))
from ee_tools import initialize_ee_with_credentials, get_s2_cloud_masked_collection


def load_mgrs_tiles(mgrs_file):
    """Load MGRS tiles from geojson file."""
    try:
        gdf = gpd.read_parquet(mgrs_file)
        return gdf
    except Exception as e:
        raise ValueError(f"Failed to load MGRS tiles from {mgrs_file}: {e}")


def find_mgrs_tile(gdf, mgrs_code):
    """Find MGRS tile by code and return geometry and CRS info."""
    tile_row = gdf[gdf['mgrs_id'] == mgrs_code]
    
    if tile_row.empty:
        raise ValueError(f"MGRS tile '{mgrs_code}' not found in the dataset")
    
    tile_info = tile_row.iloc[0]
    
    return {
        'geometry': tile_info.geometry,
        'epsg_code': tile_info.epsg
    }


def geometry_to_ee_feature(geometry):
    """Convert shapely geometry to Earth Engine geometry."""
    geom_dict = json.loads(gpd.GeoSeries([geometry]).to_json())
    coords = geom_dict['features'][0]['geometry']['coordinates']
    return ee.Geometry.Polygon(coords)


def create_s2_composite(aoi_geometry, start_date, end_date, clear_threshold=0.80):
    """Create cloud-masked Sentinel-2 composite for the area of interest."""
    
    # Get cloud-masked collection
    collection = get_s2_cloud_masked_collection(
        aoi_geometry, 
        start_date, 
        end_date, 
        clear_threshold
    )
    
    # Define the bands to export
    # Using the most commonly needed Sentinel-2 bands
    bands_to_export = [
        'B2',   # Blue
        'B3',   # Green  
        'B4',   # Red
        'B5',   # Red Edge 1
        'B6',   # Red Edge 2
        'B7',   # Red Edge 3
        'B8',   # NIR
        'B8A',  # NIR Narrow
        'B9',   # Water Vapour
        'B11',  # SWIR 1
        'B12'   # SWIR 2
    ]
    
    # Create median composite
    composite = collection.select(bands_to_export).median()
    return composite, bands_to_export


def export_band_to_drive(
        image, band_name, mgrs_code, output_folder, crs, start_date, end_date, geometry, scale=10):
    """Export a single band to Google Drive as GeoTIFF."""
    
    # Select the specific band
    band_image = image.select([band_name])
    
    # Use native resolution for 20m bands
    native_20m_bands = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    native_60m_bands = ['B9']
    export_scale = 60 if band_name in native_60m_bands else 20 if band_name in native_20m_bands else scale
    
    # Convert to appropriate data type to reduce file size
    band_image = band_image.toInt16()  # Scale and convert to int16
    
    # Create export task
    task = ee.batch.Export.image.toDrive(
        image=band_image,
        description=f'{mgrs_code}_{band_name}_{start_date}_{end_date}',
        folder=output_folder,
        fileNamePrefix=f'{mgrs_code}_{band_name}_{start_date}_{end_date}',
        scale=export_scale,
        region=geometry,  # Clip to exact tile boundary
        crs=f'EPSG:{crs}',
        maxPixels=1e9,  # Reduced from 1e13
        fileFormat='GeoTIFF'
    )
    
    return task


def main():
    parser = argparse.ArgumentParser(
        description='Export cloud-masked Sentinel-2 composite bands for MGRS tile to Google Drive'
    )
    parser.add_argument(
        '--mgrs-code',
        default='49MEM', 
        help='MGRS tile code (e.g., "13TDE")'
    )
    parser.add_argument(
        '--output_folder',
        default='MGRS_s2',
        help='Google Drive folder name for output files'
    )
    parser.add_argument(
        '--start-date',
        default='2024-01-01',
        help='Start date for composite (YYYY-MM-DD format, default: 2024-01-01)'
    )
    parser.add_argument(
        '--end-date', 
        default='2025-01-01',
        help='End date for composite (YYYY-MM-DD format, default: 2025-01-01)'
    )
    parser.add_argument(
        '--clear-threshold',
        type=float,
        default=0.80,
        help='CloudScore+ clear threshold (0-1, default: 0.80)'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        help='Export scale in meters (default: 10)'
    )
    parser.add_argument(
        '--mgrs-file',
        default='geometries/mgrs_tiles.parquet',
        help='Path to MGRS geojson file (default: geometries/mgrs_tiles.parquet)'
    )
    
    args = parser.parse_args()
    
    # Initialize Earth Engine
    if not initialize_ee_with_credentials():
        print("❌ Failed to initialize Earth Engine. Exiting.")
        return 1
    
    # Set default MGRS geojson path if not provided
    if args.mgrs_file is None:
        script_dir = Path(__file__).parent.parent.parent
        args.mgrs_file = script_dir / 'geometries' / 'mgrs_tiles.parquet'
    
    try:
        print(f"🔍 Loading MGRS tiles from {args.mgrs_file}")
        mgrs_gdf = load_mgrs_tiles(args.mgrs_file)
        
        print(f"🎯 Finding MGRS tile: {args.mgrs_code}")
        tile_info = find_mgrs_tile(mgrs_gdf, args.mgrs_code)
        
        print(f"📍 Found tile with EPSG:{tile_info['epsg_code']}")
        
        # Convert geometry to Earth Engine
        ee_geometry = geometry_to_ee_feature(tile_info['geometry'])
        
        print(f"🛰️  Creating Sentinel-2 composite from {args.start_date} to {args.end_date}")
        print(f"☁️  Using CloudScore+ threshold: {args.clear_threshold}")
        
        # Create composite
        composite, bands = create_s2_composite(
            ee_geometry,
            args.start_date,
            args.end_date,
            args.clear_threshold
        )
        
        print(f"📤 Exporting {len(bands)} bands to Google Drive folder: {args.output_folder}")
        
        # Export each band individually
        tasks = []
        for band_name in bands:
            task = export_band_to_drive(
                composite,
                band_name, 
                args.mgrs_code,
                args.output_folder,
                tile_info['epsg_code'],
                args.start_date,
                args.end_date,
                ee_geometry,
                args.scale
            )
            tasks.append(task)
            print(f"   📁 Queued export: {args.mgrs_code}_{band_name}.tif")
        
        # Start all export tasks
        for task in tasks:
            task.start()
        
        print(f"✅ Started {len(tasks)} export tasks")
        print("🔄 Monitor progress at: https://code.earthengine.google.com/tasks")
        print("📋 Export details:")
        print(f"   MGRS Code: {args.mgrs_code}")
        print(f"   Output Folder: {args.output_folder}")
        print(f"   Date Range: {args.start_date} to {args.end_date}")
        print(f"   CRS: EPSG:{tile_info['epsg_code']}")
        print(f"   Scale: {args.scale}m")
        print(f"   Bands: {', '.join(bands)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 