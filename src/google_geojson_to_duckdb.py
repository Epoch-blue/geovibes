import argparse
import logging
import pathlib
import sys
from typing import List, Set, Optional, Dict

import geopandas as gpd
import duckdb
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm
from joblib import Parallel, delayed
import tempfile
import shutil

import os

def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def _load_region(path: str) -> gpd.GeoSeries:
    """Load a vector file and return a single unified geometry in WGS84."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("No geometries found in input file")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")

def _get_utm_crs(lon: float, lat: float) -> str:
    """Get appropriate UTM CRS for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"

def get_intersecting_mgrs_ids(roi_path: str, mgrs_reference_path: str) -> Dict[str, str]:
    """Find MGRS tiles that intersect with the given ROI and return a dict of tile_id: epsg_code."""
    logging.info("Loading ROI and MGRS reference file...")
    region_gs = _load_region(roi_path)
    mgrs_gdf = gpd.read_parquet(mgrs_reference_path)

    if mgrs_gdf.crs is None or mgrs_gdf.crs.to_epsg() != 4326:
        mgrs_gdf = mgrs_gdf.to_crs(4326)

    logging.info("Finding intersecting MGRS tiles...")
    intersecting_tiles = gpd.sjoin(
        mgrs_gdf, gpd.GeoDataFrame(geometry=region_gs), how="inner", predicate="intersects"
    )

    tile_id_col = "mgrs_id"
    epsg_col = "epsg"
    if tile_id_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have a '{tile_id_col}' column.")
    if epsg_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have an '{epsg_col}' column.")

    return dict(zip(intersecting_tiles[tile_id_col], intersecting_tiles[epsg_col]))

def process_and_save_geojson(gcs_path: str, epsg_code: str, output_dir: str) -> Optional[str]:
    """
    Reads a GeoJSON from GCS, processes it, and saves it as a local GeoParquet file.
    """
    try:
        gdf = gpd.read_file(gcs_path).drop(columns=["id"])
        if gdf.empty:
            return None

        # Calculate accurate centroids in the tile's specific UTM projection
        centroids = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                centroids.append(None)
                continue
            
            # Use the provided EPSG code for the projection
            centroid_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(f"EPSG:{epsg_code}").centroid
            centroid_wgs84 = centroid_utm.to_crs("EPSG:4326").iloc[0]
            centroids.append(centroid_wgs84)

        gdf['geometry'] = centroids
        gdf = gdf[gdf.geometry.notna()]
        band_names = [f'A{i:02d}' for i in range(64)]
        gdf['embedding'] = gdf[band_names].values.tolist()
        final_cols = ['tile_id', 'geometry', 'embedding']
        return gdf[final_cols]
    
    except Exception as e:
        logging.warning(f"Failed to process {gcs_path}: {e}")
        return None

def create_duckdb_index(parquet_files: List[str], output_file: str, metric: str):
    """Creates a DuckDB database with HNSW and spatial indexes from local parquet files."""
    logging.info(f"Creating DuckDB database at {output_file} with {metric} metric...")
    
    con = duckdb.connect(database=output_file)
    
    # Use glob to read all parquet files from the directory
    parquet_paths_str = "['" + "','".join(parquet_files) + "']"
    
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Create the final table
    con.execute(f"""
    CREATE OR REPLACE TABLE geo_embeddings AS
    SELECT
        tile_id AS id,
        ST_GeomFromWKB(geometry) AS geometry,
        CAST(embedding AS FLOAT[64]) AS embedding
    FROM read_parquet({parquet_paths_str})
    """)
    
    logging.info("Creating HNSW index...")
    con.execute(f"CREATE INDEX hnsw_idx ON geo_embeddings USING HNSW (embedding) WITH (metric = '{metric}');")
    
    logging.info("Creating R-Tree spatial index...")
    con.execute("CREATE INDEX rtree_idx ON geo_embeddings USING RTREE (geometry);")
    
    con.close()
    logging.info(f"Database created successfully at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Google Satellite GeoJSON files and create a DuckDB index.")
    parser.add_argument("roi_file", help="Path to the ROI file (e.g., aoi.geojson).")
    parser.add_argument("output_dir", help="Directory to save output files.")
    parser.add_argument("output_db_file", help="Path for the output DuckDB database file.")
    parser.add_argument("--mgrs_reference_file", default="/Users/christopherren/geovibes/geometries/mgrs_tiles.parquet", help="Path to the MGRS grid reference file.")
    parser.add_argument("--gcs_bucket", default="geovibes", help="GCS bucket to use for the embeddings.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers for processing files.")
    args = parser.parse_args()

    setup_logging()
    
    # Create output directory if it doesn't exist
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for intermediate parquet files
    temp_dir = tempfile.mkdtemp(dir=args.output_dir)
    
    try:
        # 1. Find intersecting MGRS tiles and their EPSG codes
        mgrs_epsg_map = get_intersecting_mgrs_ids(args.roi_file, args.mgrs_reference_file)
        if not mgrs_epsg_map:
            logging.info("No intersecting MGRS tiles found for the given ROI.")
            return

        # 2. Build GCS paths and prepare arguments for parallel processing
        tasks = [
            (
                f"gs://{args.gcs_bucket}/embeddings/google_satellite_v1/25_0_10/{mgrs_id}_2024.geojson",
                epsg_code,
                temp_dir,
            )
            for mgrs_id, epsg_code in mgrs_epsg_map.items()
        ]
        logging.info(f"Constructed {len(tasks)} tasks to process.")

        # 3. & 4. Read, process, and save GeoJSONs in parallel
        processed_files = Parallel(n_jobs=args.workers, verbose=20)(
            delayed(process_and_save_geojson)(*task)
            for task in tasks
        )
        
        local_parquet_files = [f for f in processed_files if f is not None]
        if not local_parquet_files:
            logging.error("No data could be processed from the generated GCS paths.")
            return
        
        # 5. Create DuckDB index from the local parquet files
        create_duckdb_index(local_parquet_files, args.output_db_file, args.metric)
        
    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        logging.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main() 