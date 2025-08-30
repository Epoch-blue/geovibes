#!/usr/bin/env python3
"""
Geovibes Data Preparation Script

Downloads geometries and model databases from S3 based on manifest.csv.
Provides interactive selection interface for choosing which databases to download.
"""

import os
import sys
import csv
import tempfile
import tarfile
import shutil
import requests
import termios
import tty
import glob
from typing import List, Dict, Set
from tqdm import tqdm


def parse_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """Parse the manifest CSV file."""
    manifest_data = []

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            manifest_data.append(row)

    return manifest_data


def get_unique_regions(manifest_data: List[Dict[str, str]]) -> Set[str]:
    """Extract unique regions from manifest data."""
    return {row["region"] for row in manifest_data}


def s3_to_https_url(s3_url: str) -> str:
    """Convert S3 URL to HTTPS URL."""
    # Remove s3:// prefix and convert to https
    if s3_url.startswith("s3://"):
        path = s3_url[5:]  # Remove 's3://'
        return f"https://s3.us-west-2.amazonaws.com/{path}"
    return s3_url


def download_file_with_progress(url: str, local_path: str) -> bool:
    """Download a file with progress bar using tqdm."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        filename = os.path.basename(local_path)

        with open(local_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                ascii=True,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✓ {filename} downloaded successfully")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def download_geometry(region: str, geometries_dir: str) -> bool:
    """Download geometry file for a region."""
    geometry_url = f"https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/geovibes/geometries/{region}.geojson"
    local_path = os.path.join(geometries_dir, f"{region}.geojson")

    # Skip if file already exists
    if os.path.exists(local_path):
        print(f"Geometry for {region} already exists, skipping...")
        return True

    print(f"Downloading geometry for {region}...")
    return download_file_with_progress(geometry_url, local_path)


def extract_tar_gz(tar_path: str, extract_to: str) -> bool:
    """Extract tar.gz file to specified directory."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def download_and_extract_database(
    model_data: Dict[str, str], local_databases_dir: str
) -> bool:
    """Download and extract a model database."""
    model_name = model_data["model_name"]
    s3_path = model_data["model_path"]
    https_url = s3_to_https_url(s3_path)

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download tar.gz to temp directory
        temp_tar_path = os.path.join(temp_dir, f"{model_name}.tar.gz")

        print(f"Downloading database: {model_name}")
        if not download_file_with_progress(https_url, temp_tar_path):
            return False

        # Extract to temp directory
        print(f"Extracting {model_name}...")
        temp_extract_dir = os.path.join(temp_dir, "extracted")
        if not extract_tar_gz(temp_tar_path, temp_extract_dir):
            return False

        # Move all files from extracted contents to local_databases (flatten structure)

        # Create the model directory
        os.makedirs(local_databases_dir, exist_ok=True)

        # Find all files recursively using glob and move them to the model directory
        all_files = glob.glob(os.path.join(temp_extract_dir, "**", "*"), recursive=True)
        files_to_move = [f for f in all_files if os.path.isfile(f)]

        for src_file in files_to_move:
            filename = os.path.basename(src_file)
            dst_file = os.path.join(local_databases_dir, filename)

            # Handle filename conflicts by adding a counter
            counter = 1
            original_dst = dst_file
            while os.path.exists(dst_file):
                name, ext = os.path.splitext(original_dst)
                dst_file = f"{name}_{counter}{ext}"
                counter += 1

            shutil.move(src_file, dst_file)

        print(f"Database {model_name} extracted to {local_databases_dir}")
        return True


def get_key() -> str:
    """Get a single key press without pressing enter."""
    if os.name == "nt":  # Windows
        import msvcrt

        return msvcrt.getch().decode("utf-8")
    else:  # Unix/Linux/macOS
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if ch == "\x1b":  # ESC sequence
                ch += sys.stdin.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def display_selection_menu(manifest_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Display interactive selection menu for databases."""
    selected = [False] * len(manifest_data)
    current_index = 0

    def print_menu():
        os.system("clear" if os.name == "posix" else "cls")
        print("Geovibes Data Download - Select Databases")
        print(
            "Use UP/DOWN arrows and SPACE to select/deselect, ENTER to confirm, 'q' to quit"
        )
        print("=" * 80)

        # Column headers
        print(f"   {'Status':<8} {'Region':<15} | {'Model Name'}")
        print("-" * 80)

        for i, row in enumerate(manifest_data):
            marker = "[X]" if selected[i] else "[ ]"
            cursor = ">" if i == current_index else " "
            print(f"{cursor} {marker:<8} {row['region']:<15} | {row['model_name']}")

        print("\n" + "=" * 80)
        print(f"Selected: {sum(selected)} databases")
        print("Controls: ↑/↓ navigate, SPACE select/deselect, ENTER confirm, 'q' quit")

    while True:
        print_menu()

        try:
            key = get_key()

            if key == "q" or key == "Q":
                print("\nExiting...")
                sys.exit(0)
            elif key == "\x1b[A":  # Up arrow
                current_index = max(0, current_index - 1)
            elif key == "\x1b[B":  # Down arrow
                current_index = min(len(manifest_data) - 1, current_index + 1)
            elif key == " ":  # Space to toggle selection
                selected[current_index] = not selected[current_index]
            elif key == "\r" or key == "\n":  # Enter to confirm
                break
            elif key.isdigit():  # Number keys for quick selection
                idx = int(key)
                if 0 <= idx < len(manifest_data):
                    selected[idx] = not selected[idx]
            elif key == "a" or key == "A":  # 'a' for all
                selected = [True] * len(manifest_data)
            elif key == "n" or key == "N":  # 'n' for none
                selected = [False] * len(manifest_data)

        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    # Show final selection
    selected_items = [
        row for row, is_selected in zip(manifest_data, selected) if is_selected
    ]

    if selected_items:
        print(f"\nFinal selection ({len(selected_items)} databases):")
        for i, row in enumerate(selected_items):
            print(f"  {i + 1}. {row['region']} | {row['model_name']}")

    return selected_items


def main():
    """Main function to orchestrate the download process."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(script_dir, "geovibes", "manifest.csv")
    geometries_dir = os.path.join(script_dir, "geometries")
    local_databases_dir = os.path.join(script_dir, "local_databases")

    # Create directories if they don't exist
    os.makedirs(geometries_dir, exist_ok=True)
    os.makedirs(local_databases_dir, exist_ok=True)

    print("Geovibes Data Preparation Script")
    print("=" * 40)

    # Parse manifest
    print("Parsing manifest...")
    manifest_data = parse_manifest(manifest_path)
    print(f"Found {len(manifest_data)} database entries")

    # Display selection menu
    selected_databases = display_selection_menu(manifest_data)

    if not selected_databases:
        print("No databases selected. Exiting...")
        return

    print(f"\nSelected {len(selected_databases)} databases for download")

    # Get unique regions from selected databases
    selected_regions = get_unique_regions(selected_databases)

    # Download geometries for selected regions
    print(f"\nDownloading geometries for {len(selected_regions)} regions...")
    geometry_success = []
    for region in selected_regions:
        success = download_geometry(region, geometries_dir)
        geometry_success.append((region, success))

    # Download and extract selected databases
    print(f"\nDownloading and extracting {len(selected_databases)} databases...")
    database_success = []
    for db_data in selected_databases:
        success = download_and_extract_database(db_data, local_databases_dir)
        database_success.append((db_data["model_name"], success))

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print("\nGeometries:")
    for region, success in geometry_success:
        status = "✓" if success else "✗"
        print(f"  {status} {region}.geojson")

    print("\nDatabases:")
    for model_name, success in database_success:
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")

    successful_geometries = sum(1 for _, success in geometry_success if success)
    successful_databases = sum(1 for _, success in database_success if success)

    print(
        f"\nCompleted: {successful_geometries}/{len(geometry_success)} geometries, "
        f"{successful_databases}/{len(database_success)} databases"
    )

    if successful_geometries == len(geometry_success) and successful_databases == len(
        database_success
    ):
        print("All downloads completed successfully!")
    else:
        print("Some downloads failed. Check the output above for details.")


if __name__ == "__main__":
    main()
