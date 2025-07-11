import os
import warnings
from argparse import ArgumentParser
from typing import List, Tuple
import multiprocessing as mp
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.transform
import torch
import torch.nn as nn
from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
import kornia as K
from torch.ao.quantization import quantize_dynamic

import sys
sys.path.append('/home/christopherren/geovibes')
from src.cog.geodataset_optimized import GeoDataFrameDatasetOptimized


# Optimized pure PyTorch rescale intensity
class RescaleIntensityOptimized(nn.Module):
    """Rescale intensity values using pure PyTorch operations."""
    
    def __init__(self, out_min_max: Tuple = (0, 1), percentiles: Tuple = (2, 98)):
        super().__init__()
        self.out_min, self.out_max = out_min_max
        self.percentiles = torch.tensor(percentiles, dtype=torch.float32)
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Handle different tensor shapes
        original_shape = tensor.shape
        
        if len(original_shape) == 3:
            # [C, H, W] -> [1, C, H, W]
            tensor = tensor.unsqueeze(0)
        elif len(original_shape) == 5:
            # [B, C, 1, H, W] -> [B, C, H, W]
            tensor = tensor.squeeze(2)
        
        batch_size, channels = tensor.shape[:2]
        
        # Flatten spatial dimensions for percentile calculation
        tensor_flat = tensor.view(batch_size, channels, -1)
        
        # Calculate percentiles per channel
        lower = torch.quantile(tensor_flat, self.percentiles[0] / 100.0, dim=2, keepdim=True)
        upper = torch.quantile(tensor_flat, self.percentiles[1] / 100.0, dim=2, keepdim=True)
        
        # Reshape back to spatial dimensions
        lower = lower.view(batch_size, channels, 1, 1)
        upper = upper.view(batch_size, channels, 1, 1)
        
        # Clamp and normalize
        tensor = torch.clamp(tensor, lower, upper)
        in_range = upper - lower
        
        # Avoid division by zero
        in_range = torch.where(in_range == 0, torch.ones_like(in_range), in_range)
        
        # Normalize to [0, 1] then scale to output range
        tensor = (tensor - lower) / in_range
        out_range = self.out_max - self.out_min
        tensor = tensor * out_range + self.out_min
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            tensor = tensor.squeeze(0)
        
        return tensor


# Optimized transform pipeline
class OptimizedTransform(nn.Module):
    """Fully vectorized transform pipeline."""
    
    def __init__(self):
        super().__init__()
        self.rescale = RescaleIntensityOptimized((0, 1), (2, 98))
        self.resize = K.geometry.Resize((224, 224))
        # Use torch transforms for better compatibility
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single sample vs batch
        is_single = len(x.shape) == 3
        if is_single:
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = self.rescale(x)
        x = self.resize(x)
        
        # Manual normalization to avoid Kornia issues
        x = (x - self.mean) / self.std
        
        if is_single:
            x = x.squeeze(0)  # Remove batch dimension
        
        return x


def worker_init_fn(worker_id):
    """Initialize worker with proper thread settings."""
    # Get total CPU count and calculate threads per worker
    cpu_count = mp.cpu_count()
    num_workers = torch.utils.data.get_worker_info().num_workers
    threads_per_worker = max(1, cpu_count // num_workers)
    
    # Set thread count for this worker
    torch.set_num_threads(threads_per_worker)
    
    # Also set OMP threads
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)


class MultiResolutionBandStacker:
    """Handle Sentinel-2 bands with different resolutions."""
    
    BAND_RESOLUTIONS = {
        'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
        'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
        'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20
    }
    
    def __init__(self, target_resolution: int = 10):
        self.target_resolution = target_resolution


def download_gcs_files(mgrs_tile_id: str, bands: List[str], date_range: str, 
                      local_dir: str, gcs_bucket: str = "geovibes") -> List[str]:
    """Download GeoTIFF files from GCS, skip if already exist."""
    local_paths = []
    os.makedirs(local_dir, exist_ok=True)
    
    # Check which files need downloading
    files_to_download = []
    for band in bands:
        local_path = os.path.join(local_dir, f"{mgrs_tile_id}_{band}_{date_range}.tif")
        if os.path.exists(local_path):
            print(f"File already exists, skipping: {local_path}")
            local_paths.append(local_path)
        else:
            files_to_download.append((band, local_path))
    
    # Download missing files
    if files_to_download:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        
        for band, local_path in files_to_download:
            blob_name = f"imagery/s2/{mgrs_tile_id}_{band}_{date_range}.tif"
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.download_to_filename(local_path)
                local_paths.append(local_path)
                print(f"Downloaded {blob_name} to {local_path}")
            else:
                print(f"Warning: {blob_name} not found in GCS")
    
    return local_paths


def create_stacked_geotiff(band_paths: List[str], output_path: str, target_resolution: int = 10) -> str:
    """Create a multi-band stacked GeoTIFF from individual band files, skip if exists."""
    if os.path.exists(output_path):
        print(f"Stacked GeoTIFF already exists, skipping: {output_path}")
        return output_path
    
    stacker = MultiResolutionBandStacker(target_resolution)
    
    with rasterio.open(band_paths[0]) as template:
        profile = template.profile.copy()
        transform = template.transform
        crs = template.crs
        width = template.width
        height = template.height
        
        if template.res[0] != target_resolution:
            scale_factor = template.res[0] / target_resolution
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            left, bottom, right, top = template.bounds
            transform = rasterio.transform.from_bounds(
                left, bottom, right, top, width, height
            )
    
    profile.update({
        'count': len(band_paths),
        'width': width,
        'height': height,
        'transform': transform,
        'dtype': 'float32',
        'compress': 'lzw',  # Add compression
        'tiled': True,      # Enable tiling for better access patterns
        'blockxsize': 512,
        'blockysize': 512
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band_path in enumerate(band_paths):
            with rasterio.open(band_path) as src:
                if src.res[0] != target_resolution:
                    scale_factor = src.res[0] / target_resolution
                    new_width = int(src.width * scale_factor)
                    new_height = int(src.height * scale_factor)
                    data = src.read(1, out_shape=(new_height, new_width), 
                                  resampling=rasterio.enums.Resampling.bilinear)
                else:
                    data = src.read(1)
                dst.write(data.astype('float32'), i + 1)
    
    return output_path


def optimize_model(model: nn.Module, example_input: torch.Tensor, enable_compile: bool = True) -> nn.Module:
    """Optimize model for CPU inference."""
    model.eval()
    
    # Only try torch.compile if explicitly enabled and available
    if enable_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            # Use reduce-overhead mode for better compatibility
            model = torch.compile(model, mode="reduce-overhead")
            # Warm up with inference mode
            with torch.inference_mode():
                _ = model(example_input)
            print("Model compilation successful")
            return model
        except Exception as e:
            print(f"torch.compile failed: {e}, falling back to eager mode")
    
    # Return original model without TorchScript for better reliability
    print("Using eager mode (no compilation)")
    return model


def main(
    mgrs_tile_id: str,
    bands: List[str],
    date_range: str,
    local_dir: str,
    tiles_file: str,
    output_path: str,
    model_name: str = "resnet18",
    batch_size: int = 64,  # Increased default
    num_workers: int = 12,  # Increased default
    enable_quantization: bool = True,
    optimize_batch_size: bool = True,
    enable_compile: bool = True
) -> None:
    """Generate embeddings for Sentinel-2 imagery with CPU optimizations."""
    
    # Set global thread settings
    cpu_count = mp.cpu_count()
    print(f"System has {cpu_count} CPU cores")
    
    # Reserve some threads for main process
    main_threads = max(1, cpu_count - num_workers)
    torch.set_num_threads(main_threads)
    
    print(f"Processing MGRS tile: {mgrs_tile_id}")
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        print("Skipping processing. Delete the file to regenerate.")
        return
    
    print("Downloading GeoTIFF files from GCS...")
    band_paths = download_gcs_files(mgrs_tile_id, bands, date_range, local_dir)
    
    if not band_paths:
        raise ValueError(f"No bands found for MGRS tile {mgrs_tile_id}")
    
    stacked_path = os.path.join(local_dir, f"{mgrs_tile_id}_stacked.tif")
    print(f"Creating stacked GeoTIFF: {stacked_path}")
    create_stacked_geotiff(band_paths, stacked_path)
    
    print(f"Loading tiles from: {tiles_file}")
    
    # Handle GCS paths
    if tiles_file.startswith('gs://'):
        # Download from GCS first
        import tempfile
        
        # Parse GCS path
        gcs_parts = tiles_file.replace('gs://', '').split('/')
        bucket_name = gcs_parts[0]
        blob_path = '/'.join(gcs_parts[1:])
        
        # Download to temporary file
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        with tempfile.NamedTemporaryFile(suffix='.zip' if tiles_file.endswith('.zip') else '', delete=False) as tmp_file:
            blob.download_to_filename(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Read from temporary file
        if tiles_file.endswith('.zip'):
            tiles_gdf = gpd.read_file(f"zip://{tmp_path}")
        else:
            tiles_gdf = gpd.read_file(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
    else:
        # Local file
        if tiles_file.endswith('.zip'):
            tiles_gdf = gpd.read_file(f"zip://{tiles_file}")
        else:
            tiles_gdf = gpd.read_file(tiles_file)
    
    print(f"Loaded {len(tiles_gdf)} polygons from shapefile")
    
    # Ensure tiles_gdf has proper CRS
    with rasterio.open(stacked_path) as src:
        raster_crs = src.crs
    
    if tiles_gdf.crs != raster_crs:
        print(f"Converting tiles from {tiles_gdf.crs} to {raster_crs}")
        tiles_gdf = tiles_gdf.to_crs(raster_crs)
    
    # Create optimized transform
    transform_pipeline = OptimizedTransform()
    
    print("Creating GeoDataFrameDataset...")
    dataset = GeoDataFrameDatasetOptimized(
        path=stacked_path,
        geometries_gdf=tiles_gdf,
        transforms=transform_pipeline,
        use_memmap=True,
        cache_windows=True
    )
    
    # Optimize batch size if requested
    if optimize_batch_size:
        # Simple heuristic: aim for ~2GB memory usage per batch
        # ResNet18 embedding dim is 512, float32 = 4 bytes
        # Image size: 3 * 224 * 224 * 4 = ~600KB
        # Total per sample: ~600KB + 2KB = ~602KB
        # For 2GB: 2GB / 602KB ≈ 3400 samples
        # But account for worker memory overhead
        optimal_batch = min(128, len(dataset) // (num_workers * 2))
        batch_size = max(batch_size, optimal_batch)
        print(f"Optimized batch size: {batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,  # No CUDA
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )
    
    print(f"Loading {model_name} model...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    # Create example input for optimization
    example_input = torch.randn(1, 3, 224, 224)
    
    # Apply quantization if enabled
    if enable_quantization:
        print("Applying INT8 quantization...")
        model = quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv2d},  # Quantize linear and conv layers
            dtype=torch.qint8
        )
    
    # Optimize model (compile or eager mode)
    model = optimize_model(model, example_input, enable_compile=enable_compile)
    model.eval()
    
    print("Generating embeddings...")
    embeddings = []
    tile_ids = []
    geometry_indices = []
    
    # Use inference mode for maximum performance
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing tiles"):
            images = batch['image']
            batch_tile_ids = batch['tile_id']
            batch_geom_indices = batch['geometry_index']
            
            # CPU inference
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.numpy())
            
            # Handle both tensor and list cases for tile_ids
            if isinstance(batch_tile_ids, torch.Tensor):
                tile_ids.extend(batch_tile_ids.numpy().tolist())
            else:
                tile_ids.extend(batch_tile_ids)
                
            geometry_indices.extend(batch_geom_indices.numpy().tolist())
    
    embeddings = np.vstack(embeddings)
    
    print("Creating output GeoDataFrame...")
    
    # Vectorized result creation
    results_data = {
        'tile_id': [],
        'embedding': [],
        'geometry': [],
        'mgrs_tile_id': []
    }
    
    # Pre-allocate lists
    n_results = len(embeddings)
    results_data['tile_id'] = [None] * n_results
    results_data['embedding'] = [None] * n_results
    results_data['geometry'] = [None] * n_results
    results_data['mgrs_tile_id'] = [mgrs_tile_id] * n_results
    
    # Get additional columns
    extra_cols = [col for col in tiles_gdf.columns if col not in ['geometry', 'tile_id']]
    for col in extra_cols:
        results_data[col] = [None] * n_results
    
    # Fill results efficiently
    for i, (embedding, tile_id, geom_idx) in enumerate(zip(embeddings, tile_ids, geometry_indices)):
        original_row = tiles_gdf.iloc[geom_idx]
        results_data['tile_id'][i] = original_row.get('tile_id', tile_id)
        results_data['embedding'][i] = embedding.tolist()
        results_data['geometry'][i] = original_row.geometry
        
        for col in extra_cols:
            results_data[col][i] = original_row[col]
    
    results_gdf = gpd.GeoDataFrame(results_data, crs=tiles_gdf.crs)
    
    # Compute centroids in UTM if not already in UTM
    if results_gdf.crs and not results_gdf.crs.is_projected:
        # Convert to appropriate UTM for centroid calculation
        sample_centroid = results_gdf.geometry.iloc[0].centroid
        lon = sample_centroid.x
        utm_zone = int((lon + 180) / 6) + 1
        if utm_zone > 60:
            utm_zone = 60
        utm_crs = f"EPSG:326{utm_zone:02d}"
        
        results_utm = results_gdf.to_crs(utm_crs)
        results_utm['centroid_utm'] = results_utm.geometry.centroid
        results_utm['centroid_wgs84'] = results_utm['centroid_utm'].to_crs("EPSG:4326")
        results_gdf = results_utm.to_crs(tiles_gdf.crs)
        results_gdf['centroid_utm'] = results_utm['centroid_utm']
        results_gdf['centroid_wgs84'] = results_utm['centroid_wgs84']
    else:
        # Already in projected CRS, just compute centroids
        results_gdf['centroid_utm'] = results_gdf.geometry.centroid
        results_gdf['centroid_wgs84'] = results_gdf['centroid_utm'].to_crs("EPSG:4326")
    
    print(f"Saving embeddings to: {output_path}")
    results_gdf.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"Generated {len(embeddings)} embeddings for {mgrs_tile_id}")
    print(f"Output saved to: {output_path}")
    print(f"Keeping temporary files in: {local_dir}")
    
    # Verify output matches input
    if len(results_gdf) != len(tiles_gdf):
        print(f"WARNING: Output has {len(results_gdf)} rows but input had {len(tiles_gdf)} rows")
    else:
        print(f"✓ Output has same number of rows as input: {len(results_gdf)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('mgrs_tile_id', type=str, help='MGRS tile ID (e.g., 16RCA)')
    parser.add_argument('--bands', type=str, nargs='+', 
                       default=['B04', 'B03', 'B02'], help='Sentinel-2 bands to process')
    parser.add_argument('--date_range', type=str, 
                       default='2024-01-01_2025-01-01', help='Date range string')
    parser.add_argument('--local_dir', type=str, required=True, 
                       help='Local directory for temporary files')
    parser.add_argument('--tiles_file', type=str, required=True, 
                       help='Path to tiles geojson/geoparquet/shapefile with polygons')
    parser.add_argument('--output_path', type=str, required=True, 
                       help='Output parquet file path')
    parser.add_argument('--model_name', type=str, default='resnet18', 
                       help='Model name for timm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--no_quantization', action='store_true', 
                       help='Disable INT8 quantization')
    parser.add_argument('--no_optimize_batch', action='store_true',
                       help='Disable automatic batch size optimization')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile optimization')
    
    args = parser.parse_args()
    
    main(
        mgrs_tile_id=args.mgrs_tile_id,
        bands=args.bands,
        date_range=args.date_range,
        local_dir=args.local_dir,
        tiles_file=args.tiles_file,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enable_quantization=not args.no_quantization,
        optimize_batch_size=not args.no_optimize_batch,
        enable_compile=not args.no_compile
    )