import logging
import os
import pathlib
from typing import Optional

import fsspec
from joblib import Parallel, delayed


def get_cloud_protocol(path: str) -> Optional[str]:
    """Returns 's3' or 'gs' if the path is a cloud path, otherwise None."""
    if path.startswith("s3://"):
        return "s3"
    if path.startswith("gs://"):
        return "gs"
    return None


def list_cloud_parquet_files(cloud_path: str) -> list[str]:
    """List all parquet files in a cloud directory (GCS or S3)."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")
    if not cloud_path.endswith('/'):
        cloud_path += '/'
    if protocol == "s3":
        endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
        fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
    else:
        fs = fsspec.filesystem(protocol)
    return [f"{protocol}://{p}" for p in fs.glob(cloud_path + "*.parquet")]



def _download_single_cloud_file(cloud_path: str, temp_dir: str) -> Optional[str]:
    """Download one cloud file to a temp directory and return local path."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        logging.error(f"Invalid cloud path provided to worker: {cloud_path}")
        return None
    local_filename = os.path.join(temp_dir, os.path.basename(cloud_path))
    if os.path.exists(local_filename):
        return local_filename
    try:
        if protocol == "s3":
            endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
            fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
        else:
            fs = fsspec.filesystem(protocol)
        fs.get(cloud_path, local_filename)
    except Exception as e:
        logging.error(f"Failed to download {cloud_path}: {e}")
        return None
    return local_filename



def download_cloud_files(cloud_paths: list[str], temp_dir: str) -> list[str]:
    """Download parquet files from cloud to a temporary directory in parallel."""
    local_paths = Parallel(n_jobs=-1, prefer="threads", verbose=10)(
        delayed(_download_single_cloud_file)(cloud_path, temp_dir)
        for cloud_path in cloud_paths
    )
    return [path for path in local_paths if path is not None]



def find_embedding_files_for_mgrs_ids(mgrs_ids: list[str], embedding_dir: str) -> list[str]:
    """Find parquet files in embedding directory that contain the specified MGRS IDs."""
    found_files: list[str] = []
    if get_cloud_protocol(embedding_dir):
        try:
            all_parquet_files = list_cloud_parquet_files(embedding_dir)
            for mgrs_id in mgrs_ids:
                matching_files = [f for f in all_parquet_files if mgrs_id in os.path.basename(f)]
                found_files.extend(matching_files)
        except Exception as e:
            logging.error(f"Error listing cloud files: {e}")
            return []
    else:
        embedding_path = pathlib.Path(embedding_dir)
        if not embedding_path.exists():
            logging.error(f"Embedding directory does not exist: {embedding_dir}")
            return []
        for mgrs_id in mgrs_ids:
            patterns = [
                f"*{mgrs_id}*.parquet",
                f"{mgrs_id}_*.parquet",
                f"*_{mgrs_id}.parquet",
                f"*{mgrs_id}_embeddings.parquet",
            ]
            mgrs_files: list[str] = []
            for pattern in patterns:
                matches = list(embedding_path.glob(pattern))
                mgrs_files.extend([str(f) for f in matches])
            mgrs_files = list(set(mgrs_files))
            if mgrs_files:
                found_files.extend(mgrs_files)
    return list(set(found_files))
