#!/usr/bin/env python3

import argparse
import json
import logging
import os
import glob
from typing import List, Optional, Tuple
from pathlib import Path

import modal
from modal import Secret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("geotiff-embedding-batch")

image = (
    modal.Image.debian_slim()
    .apt_install("g++")
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "pandas>=1.5.0",
        "pyarrow>=10.0.0",
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "tqdm>=4.64.0",
        "numpy>=1.24.0",
        "joblib>=1.2.0",
        "kornia>=0.7.0",
    ])
    .add_local_dir("src/geotiff", "/root/geotiff")
)


def find_tileset_for_mgrs(mgrs_id: str, tiles_dir: str) -> Optional[str]:
    """Find the tileset file (zip/geoparquet/geojson) for a given MGRS ID."""
    logger.info(f"Searching for tileset for MGRS {mgrs_id} in directory: {tiles_dir}")
    
    # First, let's see what files are actually in the directory
    if os.path.exists(tiles_dir):
        all_files = os.listdir(tiles_dir)
        logger.info(f"Files in {tiles_dir}: {all_files[:10]}...")  # Show first 10 files
    else:
        logger.error(f"Directory does not exist: {tiles_dir}")
        return None
    
    # Search for files containing the MGRS ID
    patterns = [
        f"*{mgrs_id}*.zip",
        f"*{mgrs_id}*.geoparquet", 
        f"*{mgrs_id}*.geojson",
        f"*{mgrs_id}*.gpkg",
        f"*{mgrs_id}*.shp"
    ]
    
    for pattern in patterns:
        search_pattern = os.path.join(tiles_dir, pattern)
        logger.info(f"Searching with pattern: {search_pattern}")
        files = glob.glob(search_pattern)
        if files:
            logger.info(f"Found match: {files[0]}")
            return files[0]  # Return first match
    
    logger.warning(f"No tileset found for MGRS {mgrs_id} using any pattern")
    return None

@app.function(
    image=image,
    cpu=16,
    secrets=[Secret.from_name("gcs-aws-hmac-credentials")],
    timeout=86400,
    memory=16384,
    region='us',
    volumes={
        "/gcs-mount": modal.CloudBucketMount(
            bucket_name="geovibes",
            bucket_endpoint_url="https://storage.googleapis.com",
            secret=Secret.from_name("gcs-aws-hmac-credentials"),
        )
    }
)
def process_mgrs_geotiff_embeddings(
    mgrs_tile_id: str,
    tiles_dir: str,
    output_base_path: str,
    date_range: str = "2024-01-01_2024-12-31",
    bands = None,
    model_name: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 12,
    enable_quantization: bool = True,
    enable_compile: bool = False
):
    """Process a single MGRS tile using the main function from generate_geotiff_embeddings."""
    
    import sys
    import tempfile
    import os
    import zipfile
    from pathlib import Path
    import geopandas as gpd
    
    # Add local directories to path
    sys.path.insert(0, '/root')
    
    # Import from the copied modules
    from geotiff.generate_geotiff_embeddings import main
    
    if bands is None:
        bands = ["B04", "B03", "B02"]
    
    logger.info(f"Starting processing for MGRS tile: {mgrs_tile_id}")
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Output base path: {output_base_path}")
    logger.info(f"Date range: {date_range}")
    logger.info(f"Bands: {bands}")
    logger.info(f"Model: {model_name}")
    
    # Find tileset file for this MGRS ID
    tiles_gcs_dir = f"/gcs-mount/{tiles_dir.lstrip('gs://').lstrip('geovibes/')}"
    tileset_file = find_tileset_for_mgrs(mgrs_tile_id, tiles_gcs_dir)
    
    if not tileset_file:
        raise FileNotFoundError(f"No tileset file found for MGRS {mgrs_tile_id} in {tiles_gcs_dir}")
    
    logger.info(f"Found tileset: {tileset_file}")
    
    # Set up local working directory
    local_dir = f"/tmp/{mgrs_tile_id}"
    os.makedirs(local_dir, exist_ok=True)
    
    # Extract/copy tileset to local directory
    logger.info("Extracting tileset...")
    shapefile_local_path = None
    
    if tileset_file.endswith('.zip'):
        # Handle zipped shapefile
        zip_local_path = f"{local_dir}/shapefile.zip"
        os.system(f"cp '{tileset_file}' '{zip_local_path}'")
        
        # Extract zip
        with zipfile.ZipFile(zip_local_path, 'r') as zip_ref:
            zip_ref.extractall(f"{local_dir}/shapefile")
        
        # Find the .shp file
        shp_files = list(Path(f"{local_dir}/shapefile").glob("*.shp"))
        if shp_files:
            shapefile_local_path = str(shp_files[0])
        else:
            raise FileNotFoundError("No .shp file found in the zip archive")
    
    elif tileset_file.endswith('.shp'):
        # Copy all shapefile components
        shapefile_dir = f"{local_dir}/shapefile"
        os.makedirs(shapefile_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(tileset_file))[0]
        source_dir = os.path.dirname(tileset_file)
        
        extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        for ext in extensions:
            src_file = f"{source_dir}/{base_name}{ext}"
            if os.path.exists(src_file):
                dst_file = f"{shapefile_dir}/{base_name}{ext}"
                os.system(f"cp '{src_file}' '{dst_file}'")
        
        shapefile_local_path = f"{shapefile_dir}/{base_name}.shp"
    
    elif tileset_file.endswith(('.geoparquet', '.geojson', '.gpkg')):
        # Copy directly
        shapefile_local_path = f"{local_dir}/tileset{Path(tileset_file).suffix}"
        os.system(f"cp '{tileset_file}' '{shapefile_local_path}'")
    
    else:
        raise ValueError(f"Unsupported tileset format: {tileset_file}")
    
    logger.info(f"Using tileset: {shapefile_local_path}")
    
    # Set up output paths
    output_gcs_path = f"{output_base_path.rstrip('/')}/{mgrs_tile_id}_embeddings.geoparquet"
    output_local_path = f"{local_dir}/output.geoparquet"
    
    # Call the main function from generate_geotiff_embeddings
    try:
        main(
            mgrs_tile_id=mgrs_tile_id,
            bands=bands,
            date_range=date_range,
            local_dir=local_dir,
            tiles_file=shapefile_local_path,
            output_path=output_local_path,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            enable_quantization=enable_quantization,
            enable_compile=enable_compile,
            gcs_mount_path="/gcs-mount"
        )
        
        # Copy output to GCS
        logger.info("Copying output to GCS...")
        if output_gcs_path.startswith('gs://'):
            gcs_output_path = output_gcs_path[5:]  # Remove 'gs://'
            gcs_mount_output_path = f"/gcs-mount/{gcs_output_path}"
        else:
            # Assume it's a relative path within the bucket
            gcs_mount_output_path = f"/gcs-mount/{output_gcs_path}"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(gcs_mount_output_path), exist_ok=True)
        os.system(f"cp '{output_local_path}' '{gcs_mount_output_path}'")
        
        logger.info(f"Output saved to: {output_gcs_path}")
        
        # Get result info
        result_gdf = gpd.read_parquet(output_local_path)
        processed_patches = len(result_gdf)
        feature_dimension = len(result_gdf['embedding'].iloc[0]) if len(result_gdf) > 0 else 0
        
        # Cleanup local files
        os.system(f"rm -rf {local_dir}")
        
        logger.info("Processing complete!")
        return {
            "mgrs_tile_id": mgrs_tile_id,
            "processed_patches": processed_patches,
            "output_path": output_gcs_path,
            "model_used": model_name,
            "feature_dimension": feature_dimension
        }
        
    except Exception as e:
        logger.error(f"Error processing {mgrs_tile_id}: {str(e)}")
        # Cleanup on error
        os.system(f"rm -rf {local_dir}")
        raise


@app.local_entrypoint()
def main(
    config: str,
    tiles_dir: str = None,
    output_base_path: str = None,
    date_range: str = "2024-01-01_2024-12-31",
    bands = None,
    model_name: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 12,
    enable_quantization: bool = True,
    enable_compile: bool = False
):
    """Launch Modal inference jobs for multiple MGRS tiles using config file."""
    
    # Load config file
    logger.info(f"Loading config from: {config}")
    with open(config, 'r') as f:
        config_params = json.load(f)
    
    # Extract MGRS IDs from config (required)
    mgrs_ids = config_params.get("mgrs_ids")
    if not mgrs_ids:
        raise ValueError("Config file must contain 'mgrs_ids' field with list of MGRS tile IDs")
    
    # Override parameters with config values if provided
    tiles_dir = config_params.get("tiles_dir", tiles_dir)
    output_base_path = config_params.get("output_base_path", output_base_path)
    date_range = config_params.get("date_range", date_range)
    bands = config_params.get("bands", bands)
    model_name = config_params.get("model_name", model_name)
    batch_size = config_params.get("batch_size", batch_size)
    num_workers = config_params.get("num_workers", num_workers)
    enable_quantization = config_params.get("enable_quantization", enable_quantization)
    enable_compile = config_params.get("enable_compile", enable_compile)
    
    # Validate required parameters
    if not tiles_dir:
        raise ValueError("tiles_dir must be provided either in config or as command line argument")
    if not output_base_path:
        raise ValueError("output_base_path must be provided either in config or as command line argument")
    
    if bands is None:
        bands = ["B4", "B3", "B2"]
    
    logger.info(f"Launching {len(mgrs_ids)} Modal inference jobs...")
    logger.info(f"MGRS IDs: {mgrs_ids}")
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Output base path: {output_base_path}")
    
    # Create parameter tuples for spawn
    job_params = []
    for mgrs_id in mgrs_ids:
        params = {
            "mgrs_tile_id": mgrs_id,
            "tiles_dir": tiles_dir,
            "output_base_path": output_base_path,
            "date_range": date_range,
            "bands": bands,
            "model_name": model_name,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "enable_quantization": enable_quantization,
            "enable_compile": enable_compile
        }
        job_params.append(params)
    
    # Use spawn_map for batch processing
    logger.info(f"Starting spawn_map for {len(job_params)} jobs...")
    
    # Extract parameters into individual arguments for spawn_map
    results = []
    for params in job_params:
        job = process_mgrs_geotiff_embeddings.spawn(
            mgrs_tile_id=params["mgrs_tile_id"],
            tiles_dir=params["tiles_dir"],
            output_base_path=params["output_base_path"],
            date_range=params["date_range"],
            bands=params["bands"],
            model_name=params["model_name"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            enable_quantization=params["enable_quantization"],
            enable_compile=params["enable_compile"]
        )
        results.append(job)
    
    logger.info(f"Spawned {len(results)} jobs. Jobs will run asynchronously.")
    logger.info("Use --detach flag to disconnect while jobs run, or wait for completion:")
    
    # Optionally wait for all to complete
    completed_results = []
    for i, job in enumerate(results):
        mgrs_id = mgrs_ids[i]
        logger.info(f"Waiting for job {mgrs_id}...")
        try:
            result = job.get()
            completed_results.append(result)
            logger.info(f"Job {mgrs_id} completed: {result}")
        except Exception as e:
            logger.error(f"Job {mgrs_id} failed: {e}")
            completed_results.append({"mgrs_tile_id": mgrs_id, "error": str(e)})
    
    logger.info("All jobs completed!")
    return completed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process MGRS tiles for geotiff embeddings using Modal")
    
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file containing mgrs_ids and other parameters"
    )
    
    parser.add_argument(
        "--tiles-dir",
        help="GCS directory containing tileset files (overrides config)"
    )
    
    parser.add_argument(
        "--output-base-path",
        help="Base GCS path for output files (overrides config)"
    )
    
    parser.add_argument(
        "--date-range",
        default="2024-01-01_2024-12-31",
        help="Date range for Sentinel-2 imagery (default: 2024-01-01_2024-12-31)"
    )
    
    parser.add_argument(
        "--bands",
        nargs='+',
        default=["B4", "B3", "B2"],
        help="Sentinel-2 bands to use (default: B4 B3 B2)"
    )
    
    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="timm model name to use for embeddings (default: resnet18)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of DataLoader workers (default: 12)"
    )
    
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable INT8 quantization"
    )
    
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization"
    )
    
    
    args = parser.parse_args()
    
    params = {
        "config": args.config,
        "tiles_dir": args.tiles_dir,
        "output_base_path": args.output_base_path,
        "date_range": args.date_range,
        "bands": args.bands,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "enable_quantization": not args.no_quantization,
        "enable_compile": not args.no_compile
    }
    
    # Filter out None values for cleaner output
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running batch inference with parameters: {params}")
    print("Note: Run with 'modal run' for Modal execution")