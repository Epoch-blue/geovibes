#!/usr/bin/env python3
"""
Modal-based tile embedding inference script.
Processes tile images from GCS and generates embeddings using configurable timm models.
"""

import argparse
import io
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
from modal import Secret

# Core processing dependencies
import torch
import timm
import numpy as np
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mercantile
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app setup
app = modal.App("tile-embeddings")

# Define Modal image with all required dependencies
image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "timm>=0.9.0",
    "pillow>=9.0.0",
    "pandas>=1.5.0",
    "pyarrow>=10.0.0",
    "mercantile>=1.2.0",
    "tqdm>=4.64.0",
    "boto3>=1.26.0",
    "numpy>=1.24.0"
])


def parse_tile_from_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse tile coordinates from filename like 'tile_14_13292_8548.png'
    Returns (zoom, x, y) or None if parsing fails
    """
    pattern = r'tile_(\d+)_(\d+)_(\d+)\.'
    match = re.search(pattern, filename)
    
    if match:
        zoom, x, y = map(int, match.groups())
        return zoom, x, y
    else:
        logger.warning(f"Could not parse tile coordinates from: {filename}")
        return None


class TileImageDataset:
    """
    Optimized dataset for loading tile images from GCS with parallel downloads
    """
    
    def __init__(self, gcs_bucket: str, gcs_prefix: str, s3_client, transforms, max_files: Optional[int] = None, num_workers: int = 8):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.s3_client = s3_client
        self.transforms = transforms
        self.num_workers = num_workers
        self.valid_files = []
        self.tile_coords = []
        
        # List and filter image files
        self._discover_files(max_files)
        
        logger.info(f"Dataset initialized with {len(self.valid_files)} valid tile images")
        logger.info(f"Using {num_workers} workers for parallel downloads")
    
    def _discover_files(self, max_files: Optional[int] = None):
        """Discover and filter valid tile image files from GCS"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.gcs_bucket,
                Prefix=self.gcs_prefix
            )
            
            count = 0
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                            coords = parse_tile_from_filename(key)
                            if coords is not None:
                                self.valid_files.append(key)
                                self.tile_coords.append(coords)
                                count += 1
                                
                                if max_files and count >= max_files:
                                    return
                                    
        except ClientError as e:
            logger.error(f"Error listing GCS objects: {e}")
            raise
    
    def __len__(self):
        return len(self.valid_files)
    
    def _download_single_image(self, idx: int) -> Dict[str, Any]:
        """Download and process a single image"""
        try:
            key = self.valid_files[idx]
            coords = self.tile_coords[idx]
            
            # Download image from GCS
            response = self.s3_client.get_object(Bucket=self.gcs_bucket, Key=key)
            img_data = response['Body'].read()
            
            # Load and transform image
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_tensor = self.transforms(img)
            
            return {
                'image': img_tensor,
                'key': key,
                'coords': coords,
                'valid': True,
                'idx': idx
            }
            
        except Exception as e:
            logger.warning(f"Failed to load image {key}: {e}")
            return {
                'image': torch.zeros(3, 224, 224),  # Dummy tensor
                'key': self.valid_files[idx],
                'coords': self.tile_coords[idx],
                'valid': False,
                'idx': idx
            }
    
    def load_image_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """Load a batch of images from GCS using parallel downloads"""
        end_idx = min(start_idx + batch_size, len(self.valid_files))
        indices = list(range(start_idx, end_idx))
        
        # Use ThreadPoolExecutor for parallel downloads
        batch_data = [None] * len(indices)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all download tasks
            future_to_idx = {executor.submit(self._download_single_image, idx): i 
                           for i, idx in enumerate(indices)}
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                try:
                    result = future.result()
                    batch_data[result_idx] = result
                except Exception as e:
                    logger.error(f"Error in parallel download: {e}")
                    # Create dummy entry
                    orig_idx = indices[result_idx]
                    batch_data[result_idx] = {
                        'image': torch.zeros(3, 224, 224),
                        'key': self.valid_files[orig_idx],
                        'coords': self.tile_coords[orig_idx],
                        'valid': False,
                        'idx': orig_idx
                    }
        
        return batch_data


def prepare_tile_data(embeddings: np.ndarray, image_keys: List[str], tile_coords: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    """Prepare tile data with embeddings and geometries"""
    logger.info("Preparing tile data with geometries...")
    
    tile_data = []
    
    for i, (embedding, key, coords) in enumerate(zip(embeddings, image_keys, tile_coords)):
        zoom, x, y = coords
        
        # Get tile bounds using mercantile
        tile_bounds = mercantile.bounds(x, y, zoom)
        
        # Calculate tile center
        center_lon = (tile_bounds.west + tile_bounds.east) / 2
        center_lat = (tile_bounds.south + tile_bounds.north) / 2
        
        # Create WKT point geometry for tile center
        geometry_wkt = f"POINT({center_lon} {center_lat})"
        
        # Create tile ID
        tile_id = f"{zoom}_{x}_{y}"
        
        tile_data.append({
            'id': tile_id,
            'zoom': zoom,
            'x': x,
            'y': y,
            'embedding': embedding.tolist(),
            'geometry_wkt': geometry_wkt,
            'image_key': key,
            'west': tile_bounds.west,
            'south': tile_bounds.south,
            'east': tile_bounds.east,
            'north': tile_bounds.north,
            'center_lon': center_lon,
            'center_lat': center_lat
        })
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(embeddings)} tiles...")
    
    logger.info(f"Prepared {len(tile_data)} tile records")
    return tile_data


def write_parquet_to_gcs(tile_data: List[Dict[str, Any]], output_gcs_path: str, s3_client, chunk_size: int = 10000):
    """Write tile data to parquet files in GCS"""
    logger.info(f"Writing {len(tile_data)} records to {output_gcs_path}")
    
    # Parse GCS path
    if not output_gcs_path.startswith('gs://'):
        raise ValueError("Output path must start with 'gs://'")
    
    path_parts = output_gcs_path[5:].split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Split data into chunks and write multiple parquet files
    for chunk_idx in range(0, len(tile_data), chunk_size):
        chunk_data = tile_data[chunk_idx:chunk_idx + chunk_size]
        
        # Create DataFrame
        df = pd.DataFrame(chunk_data)
        
        # Convert to parquet
        table = pa.Table.from_pandas(df)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            pq.write_table(table, tmp_file.name)
            
            # Upload to GCS
            chunk_key = f"{prefix}/embeddings_chunk_{chunk_idx:06d}.parquet"
            try:
                s3_client.upload_file(tmp_file.name, bucket, chunk_key)
                logger.info(f"Uploaded chunk {chunk_idx//chunk_size + 1} to gs://{bucket}/{chunk_key}")
            finally:
                os.unlink(tmp_file.name)


@app.function(
    image=image,
    gpu="T4",
    secrets=[Secret.from_name("gcs-hmac-credentials")],
    timeout=3600,
    memory=8192
)
def process_tile_embeddings(
    input_gcs_path: str,
    output_gcs_path: str,
    model_name: str = "resnet34.a3_in1k",
    batch_size: int = 256,  # Increased from 64 to better utilize T4 GPU
    max_files: Optional[int] = None,
    chunk_size: int = 10000,
    num_workers: int = 8  # Number of parallel download threads
):
    """
    Main function to process tile embeddings using Modal.
    
    Args:
        input_gcs_path: GCS path to input tile images (e.g., gs://bucket/path/to/tiles)
        output_gcs_path: GCS path for output parquet files
        model_name: timm model name to use for embeddings
        batch_size: Batch size for processing (default: 256, optimized for T4 GPU)
        max_files: Maximum number of files to process (for testing)
        chunk_size: Number of records per parquet file
        num_workers: Number of parallel download threads (default: 8)
    """
    
    logger.info(f"Starting tile embedding processing")
    logger.info(f"Input: {input_gcs_path}")
    logger.info(f"Output: {output_gcs_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    
    # Configure S3 client for GCS
    s3_client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url="https://storage.googleapis.com",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    
    # Parse input GCS path
    if not input_gcs_path.startswith('gs://'):
        raise ValueError("Input path must start with 'gs://'")
    
    path_parts = input_gcs_path[5:].split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classifier
        )
        model = model.eval().to(device)
        
        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        logger.info(f"Model loaded. Feature dimension: {model.num_features}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    
    # Create dataset
    logger.info("Initializing dataset...")
    dataset = TileImageDataset(bucket, prefix, s3_client, transforms, max_files, num_workers)
    
    if len(dataset) == 0:
        logger.warning("No valid tile images found!")
        return
    
    # Process images in batches
    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    
    all_embeddings = []
    all_keys = []
    all_coords = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_data = dataset.load_image_batch(batch_start, batch_size)
            
            # Filter valid images
            valid_batch = [item for item in batch_data if item['valid']]
            
            if not valid_batch:
                continue
            
            # Stack images into batch tensor
            batch_images = torch.stack([item['image'] for item in valid_batch]).to(device)
            
            # Extract embeddings
            batch_embeddings = model(batch_images)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Store results
            for i, item in enumerate(valid_batch):
                all_embeddings.append(batch_embeddings[i])
                all_keys.append(item['key'])
                all_coords.append(item['coords'])
            
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    logger.info(f"Embedding extraction completed in {total_time:.1f} seconds")
    logger.info(f"Average speed: {len(all_embeddings) / total_time:.1f} images/second")
    
    # Prepare tile data with geometries
    embeddings_array = np.array(all_embeddings)
    tile_data = prepare_tile_data(embeddings_array, all_keys, all_coords)
    
    # Write results to GCS
    write_parquet_to_gcs(tile_data, output_gcs_path, s3_client, chunk_size)
    
    logger.info("Processing complete!")
    return {
        "processed_images": len(all_embeddings),
        "output_path": output_gcs_path,
        "model_used": model_name,
        "processing_time": total_time
    }


@app.local_entrypoint()
def main(
    input_gcs_path: str,
    output_gcs_path: str,
    model_name: str = "resnet34.a3_in1k",
    batch_size: int = 256,
    max_files: int = None,
    chunk_size: int = 10000,
    num_workers: int = 8,
    config: str = None
):
    """
    Local entrypoint for running the Modal function
    
    Args:
        input_gcs_path: GCS path to input tile images (e.g., gs://bucket/path/to/tiles)
        output_gcs_path: GCS path for output parquet files
        model_name: timm model name to use for embeddings (default: resnet34.a3_in1k)
        batch_size: Batch size for processing (default: 256, optimized for T4 GPU)
        max_files: Maximum number of files to process (for testing)
        chunk_size: Number of records per parquet file (default: 10000)
        num_workers: Number of parallel download threads (default: 8)
        config: Path to JSON config file with parameters
    """
    
    # Load config file if provided
    config_params = {}
    if config:
        with open(config, 'r') as f:
            config_params = json.load(f)
    
    # Override config with command line args
    params = {
        "input_gcs_path": input_gcs_path,
        "output_gcs_path": output_gcs_path,
        "model_name": model_name,
        "batch_size": batch_size,
        "max_files": max_files,
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        **config_params
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running with parameters: {params}")
    
    # Run the Modal function
    result = process_tile_embeddings.remote(**params)
    
    print(f"Processing complete! Results: {result}")


if __name__ == "__main__":
    # Fallback for direct Python execution
    parser = argparse.ArgumentParser(description="Process tile images and generate embeddings using Modal")
    
    parser.add_argument(
        "--input-gcs-path", 
        required=True,
        help="GCS path to input tile images (e.g., gs://bucket/path/to/tiles)"
    )
    
    parser.add_argument(
        "--output-gcs-path",
        required=True, 
        help="GCS path for output parquet files"
    )
    
    parser.add_argument(
        "--model-name",
        default="resnet34.a3_in1k",
        help="timm model name to use for embeddings (default: resnet34.a3_in1k)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing (default: 256, optimized for T4 GPU)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of records per parquet file (default: 10000)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to JSON config file with parameters"
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line args
    params = {
        "input_gcs_path": args.input_gcs_path,
        "output_gcs_path": args.output_gcs_path,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_files": args.max_files,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        **config
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running with parameters: {params}")
    print("Note: Run with 'modal run' for Modal execution or 'python' for direct execution")