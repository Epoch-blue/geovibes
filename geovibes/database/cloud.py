import logging
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import fsspec
from tqdm import tqdm

# Backward compatibility placeholders for older interfaces/tests that monkeypatch
Parallel = None  # type: ignore
delayed = None  # type: ignore


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
        endpoint = os.environ.get("S3_ENDPOINT_URL", "https://s3.us-west-2.amazonaws.com")
        use_anon = os.environ.get("GEOVIBES_S3_USE_ANON", "true").lower() != "false"
        fs = fsspec.filesystem(
            "s3",
            anon=use_anon,
            client_kwargs={"endpoint_url": endpoint},
        )
        matches = fs.glob(cloud_path + "*.parquet")
        return [f"{protocol}://{p}" for p in matches]
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
            endpoint = os.environ.get("S3_ENDPOINT_URL", "https://s3.us-west-2.amazonaws.com")
            use_anon = os.environ.get("GEOVIBES_S3_USE_ANON", "true").lower() != "false"
            fs = fsspec.filesystem(
                "s3",
                anon=use_anon,
                client_kwargs={"endpoint_url": endpoint},
            )
            fs.get(cloud_path, local_filename)
        else:
            fs = fsspec.filesystem(protocol)
            fs.get(cloud_path, local_filename)
    except Exception as e:
        logging.error(f"Failed to download {cloud_path}: {e}")
        return None
    return local_filename



def download_cloud_files(cloud_paths: list[str], temp_dir: str) -> list[str]:
    """Download parquet files from cloud to a temporary directory in parallel."""
    if not cloud_paths:
        return []

    max_workers = min(8, len(cloud_paths))
    local_paths: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_cloud_file, path, temp_dir): path
            for path in cloud_paths
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading cloud files",
            unit="file",
        ):
            path = futures[future]
            try:
                result = future.result()
                if result:
                    local_paths.append(result)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error(f"Failed to download {path}: {exc}")

    return local_paths



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
