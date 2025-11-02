"""
Extract satellite embeddings from Google Earth Engine using xee and xarray.

This module provides efficient methods to:
1. Resample embeddings to a specified scale using GEE
2. Extract data as xarray DataArrays
3. Convert to coordinate-based points for database ingestion
4. Stream to Parquet for large datasets
"""

from typing import Optional, Tuple, Iterator
import ee
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def initialize_earth_engine(service_account_key: Optional[str] = None) -> None:
    """
    Initialize Earth Engine with automatic authentication.
    
    Args:
        service_account_key: Optional path to Google service account key JSON file
        
    Raises:
        Exception: If authentication fails
    """
    try:
        ee.Initialize()
        print("✅ Earth Engine initialized successfully")
    except Exception as e:
        print(f"⚠️  Initial initialization failed: {e}")
        print("🔄 Attempting to authenticate...")
        
        try:
            if service_account_key:
                print(f"🔑 Using service account key: {service_account_key}")
                ee.Authenticate(service_account_key)
            else:
                ee.Authenticate()
            
            ee.Initialize()
            print("✅ Earth Engine authenticated and initialized successfully")
        except Exception as auth_error:
            print(f"❌ Authentication failed: {auth_error}")
            print("\n💡 Authentication options:")
            print("1. Run: earthengine authenticate")
            print("2. Use --service-account-key path/to/key.json")
            print("3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            raise


def load_roi_geometry(
    roi_geojson: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> ee.Geometry:
    """
    Load region of interest geometry from file or bounding box.
    
    Args:
        roi_geojson: Path to GeoJSON file defining ROI
        bbox: Bounding box as (west, south, east, north) in decimal degrees
        
    Returns:
        Earth Engine Geometry object
        
    Raises:
        ValueError: If neither or both roi_geojson and bbox are provided
    """
    if roi_geojson and bbox:
        raise ValueError("Provide either roi_geojson or bbox, not both")
    
    if not roi_geojson and not bbox:
        raise ValueError("Must provide either roi_geojson or bbox")
    
    if roi_geojson:
        import geopandas as gpd
        roi_gdf = gpd.read_file(roi_geojson)
        roi_geom = roi_gdf.geometry.iloc[0]
        roi_geojson_dict = roi_geom.__geo_interface__
        roi_ee = ee.Geometry(roi_geojson_dict)
        
        bounds = roi_geom.bounds
        print(f"🗺️  Geometry type: {roi_geom.geom_type}")
        print(f"📐 Bounds: {bounds}")
    else:
        west, south, east, north = bbox
        roi_ee = ee.Geometry.Rectangle([west, south, east, north])
        print(f"📐 Bounding box: west={west}, south={south}, east={east}, north={north}")
    
    return roi_ee


def get_alphaearth_embeddings(
    roi_ee: ee.Geometry,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    scale: int = 10
) -> ee.Image:
    """
    Retrieve and composite AlphaEarth embeddings for region.
    
    Args:
        roi_ee: Earth Engine Geometry object
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        scale: Resolution in meters
        
    Returns:
        Earth Engine Image with AlphaEarth embeddings
    """
    alphaearth_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    alphaearth_filtered = alphaearth_collection.filterDate(start_date, end_date).filterBounds(roi_ee)
    alphaearth_image = alphaearth_filtered.mosaic()
    
    band_names = alphaearth_image.bandNames().getInfo()
    print(f"🎯 Found {len(band_names)} embedding bands")
    print(f"📋 Band names: {band_names[:10]}...")
    
    return alphaearth_image


def resample_embeddings(
    image: ee.Image,
    roi_ee: ee.Geometry,
    scale: int = 10,
    resample_method: str = "bilinear"
) -> ee.Image:
    """
    Resample embeddings to specified scale using GEE.
    
    Args:
        image: Earth Engine Image with embeddings
        roi_ee: Region of interest geometry
        scale: Target resolution in meters
        resample_method: Resampling method ('bilinear', 'nearest', 'bicubic')
        
    Returns:
        Resampled Earth Engine Image
    """
    print(f"🔧 Resampling to {scale}m using {resample_method} interpolation")
    
    resampled = image.resample(resample_method).reproject(
        crs='EPSG:4326',
        scale=scale
    )
    
    return resampled


def extract_to_xarray(
    image: ee.Image,
    roi_ee: ee.Geometry,
    scale: int = 10,
    max_pixels: int = int(1e9)
) -> xr.DataArray:
    """
    Extract Earth Engine image to xarray DataArray.
    
    Args:
        image: Earth Engine Image to extract
        roi_ee: Region of interest geometry
        scale: Resolution in meters
        max_pixels: Maximum pixels to process
        
    Returns:
        xarray DataArray with image data
    """
    try:
        import xee
    except ImportError:
        raise ImportError("xee not installed. Install with: pip install xee")
    
    print(f"📊 Extracting to xarray (max_pixels={max_pixels})")
    
    data = xee.mask_and_scale(
        xee.open_rasterio(
            image,
            geometry=roi_ee,
            scale=scale,
            crs='EPSG:4326'
        )
    )
    
    return data


def pixels_to_coordinates(
    xarray_data: xr.DataArray,
    tile_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert xarray pixel data to coordinate-based points with embeddings.
    
    Each pixel becomes a point at its center coordinates, with embedding values
    from all bands stored as a single embedding column.
    
    Args:
        xarray_data: xarray DataArray from xee extraction
        tile_id: Optional identifier for the tile
        
    Returns:
        DataFrame with columns: lon, lat, embedding (array), tile_id, and individual band values
    """
    print("🔄 Converting pixels to coordinate points...")
    
    coords_x = xarray_data.x.values
    coords_y = xarray_data.y.values
    
    records = []
    
    for y_idx, lat in enumerate(coords_y):
        for x_idx, lon in enumerate(coords_x):
            pixel_data = xarray_data.isel(x=x_idx, y=y_idx).values
            
            if np.isnan(pixel_data).all():
                continue
            
            embedding_values = pixel_data[~np.isnan(pixel_data)]
            
            if len(embedding_values) == 0:
                continue
            
            record = {
                'lon': float(lon),
                'lat': float(lat),
                'embedding': embedding_values.astype(np.float32),
                'tile_id': tile_id or 'default'
            }
            
            for band_idx, value in enumerate(embedding_values):
                record[f'band_{band_idx}'] = float(value)
            
            records.append(record)
    
    df = pd.DataFrame(records)
    print(f"✅ Created {len(df)} coordinate points from xarray")
    
    return df


def stream_pixels_to_parquet(
    xarray_data: xr.DataArray,
    parquet_path: str,
    tile_id: Optional[str] = None,
    batch_size: int = 10000
) -> int:
    """
    Stream convert xarray pixels to coordinate points and write to Parquet in batches.
    
    Processes pixels in batches to avoid loading entire dataset into memory.
    
    Args:
        xarray_data: xarray DataArray from xee extraction
        parquet_path: Path to output Parquet file
        tile_id: Optional identifier for the tile
        batch_size: Number of rows per batch
        
    Returns:
        Total number of rows written
    """
    print(f"📝 Streaming pixels to Parquet: {parquet_path}")
    
    coords_x = xarray_data.x.values
    coords_y = xarray_data.y.values
    
    total_rows = 0
    all_records = []
    
    for y_idx, lat in enumerate(coords_y):
        for x_idx, lon in enumerate(coords_x):
            pixel_data = xarray_data.isel(x=x_idx, y=y_idx).values
            
            if np.isnan(pixel_data).all():
                continue
            
            embedding_values = pixel_data[~np.isnan(pixel_data)]
            
            if len(embedding_values) == 0:
                continue
            
            record = {
                'lon': float(lon),
                'lat': float(lat),
                'embedding': embedding_values.astype(np.float32),
                'tile_id': tile_id or 'default'
            }
            
            for band_idx, value in enumerate(embedding_values):
                record[f'band_{band_idx}'] = float(value)
            
            all_records.append(record)
            
            if len(all_records) >= batch_size:
                df_batch = pd.DataFrame(all_records)
                if total_rows == 0:
                    df_batch.to_parquet(parquet_path, index=False)
                else:
                    df_batch.to_parquet(parquet_path, index=False, append=True)
                
                batch_count = len(all_records)
                total_rows += batch_count
                print(f"   💾 Wrote batch: {batch_count} rows (total: {total_rows})")
                all_records = []
    
    if all_records:
        df_batch = pd.DataFrame(all_records)
        if total_rows == 0:
            df_batch.to_parquet(parquet_path, index=False)
        else:
            df_batch.to_parquet(parquet_path, index=False, append=True)
        
        batch_count = len(all_records)
        total_rows += batch_count
        print(f"   💾 Wrote final batch: {batch_count} rows (total: {total_rows})")
    
    print(f"✅ Streamed {total_rows} coordinate points to Parquet")
    
    return total_rows


def extract_embeddings_streaming_generator(
    roi_geojson: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    scale: int = 10,
    resample_scale: Optional[int] = None,
    tile_id: Optional[str] = None,
    service_account_key: Optional[str] = None,
    batch_size: int = 10000
) -> Iterator[pd.DataFrame]:
    """
    Extract embeddings as a generator of DataFrames for memory-efficient processing.
    
    Yields batches of coordinate points as they're extracted. Useful for:
    - Processing very large regions
    - Streaming directly to Parquet
    - Streaming directly to database
    
    Args:
        roi_geojson: Path to GeoJSON file defining ROI
        bbox: Bounding box as (west, south, east, north)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        scale: Initial GEE scale in meters
        resample_scale: Optional resampling scale
        tile_id: Optional identifier for the tile
        service_account_key: Optional path to service account key
        batch_size: Rows per yielded DataFrame
        
    Yields:
        DataFrames with extracted embeddings as coordinate points
    """
    initialize_earth_engine(service_account_key)
    
    roi_ee = load_roi_geometry(roi_geojson, bbox)
    
    image = get_alphaearth_embeddings(roi_ee, start_date, end_date, scale)
    
    if resample_scale and resample_scale != scale:
        image = resample_embeddings(image, roi_ee, resample_scale)
        effective_scale = resample_scale
    else:
        effective_scale = scale
    
    xarray_data = extract_to_xarray(image, roi_ee, effective_scale)
    
    coords_x = xarray_data.x.values
    coords_y = xarray_data.y.values
    
    records = []
    
    print("🔄 Yielding coordinate points in batches...")
    
    for y_idx, lat in enumerate(coords_y):
        for x_idx, lon in enumerate(coords_x):
            pixel_data = xarray_data.isel(x=x_idx, y=y_idx).values
            
            if np.isnan(pixel_data).all():
                continue
            
            embedding_values = pixel_data[~np.isnan(pixel_data)]
            
            if len(embedding_values) == 0:
                continue
            
            record = {
                'lon': float(lon),
                'lat': float(lat),
                'embedding': embedding_values.astype(np.float32),
                'tile_id': tile_id or 'default'
            }
            
            for band_idx, value in enumerate(embedding_values):
                record[f'band_{band_idx}'] = float(value)
            
            records.append(record)
            
            if len(records) >= batch_size:
                yield pd.DataFrame(records)
                records = []
    
    if records:
        yield pd.DataFrame(records)


def extract_embeddings_to_dataframe(
    roi_geojson: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    scale: int = 10,
    resample_scale: Optional[int] = None,
    tile_id: Optional[str] = None,
    service_account_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Complete pipeline to extract AlphaEarth embeddings as coordinate points.
    
    Args:
        roi_geojson: Path to GeoJSON file defining ROI
        bbox: Bounding box as (west, south, east, north)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        scale: Initial GEE scale in meters
        resample_scale: Optional resampling scale (if different from scale)
        tile_id: Optional identifier for the tile
        service_account_key: Optional path to service account key
        
    Returns:
        DataFrame with extracted embeddings as coordinate points
    """
    initialize_earth_engine(service_account_key)
    
    roi_ee = load_roi_geometry(roi_geojson, bbox)
    
    image = get_alphaearth_embeddings(roi_ee, start_date, end_date, scale)
    
    if resample_scale and resample_scale != scale:
        image = resample_embeddings(image, roi_ee, resample_scale)
        effective_scale = resample_scale
    else:
        effective_scale = scale
    
    xarray_data = extract_to_xarray(image, roi_ee, effective_scale)
    
    df = pixels_to_coordinates(xarray_data, tile_id)
    
    return df
