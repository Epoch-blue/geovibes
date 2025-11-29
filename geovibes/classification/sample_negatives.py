"""
Sample negatives from DuckDB embeddings using ESRI LULC stratification.

Adapted from ei-notebook/src/sample_negatives.py

This module samples negative training examples from the embedding database,
stratified by ESRI Global LULC classes, while ensuring negatives are
spatially separated from positive examples.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

# Earth Engine imports - optional, only needed for LULC labeling
try:
    import ee
    import geemap

    HAS_EE = True
except ImportError:
    HAS_EE = False


@dataclass
class LULCConfig:
    """Configuration for LULC-based negative sampling."""

    collection: str = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    class_mapping: Dict[int, str] = None

    def __post_init__(self):
        if self.class_mapping is None:
            # Default ESRI LULC class mapping
            self.class_mapping = {
                1: "water",
                2: "trees",
                4: "flooded_vegetation",
                5: "crops",
                7: "built_area",
                8: "bare_ground",
                9: "snow_ice",
                10: "clouds",
                11: "rangeland",
            }


@dataclass
class SamplingConfig:
    """Configuration for negative sampling."""

    samples_per_class: Dict[str, int] = None  # e.g., {"crops": 500, "trees": 500}
    buffer_meters: float = 500.0  # Min distance from positive points
    random_seed: int = 42

    def __post_init__(self):
        if self.samples_per_class is None:
            self.samples_per_class = {
                "crops": 500,
                "trees": 500,
                "built_area": 200,
                "rangeland": 300,
                "bare_ground": 200,
            }


class NegativeSampler:
    """
    Sample negative training examples from DuckDB embeddings.

    Uses ESRI Global LULC to stratify samples by landcover class,
    and filters out samples too close to positive examples.
    """

    def __init__(
        self,
        duckdb_connection,
        lulc_config: Optional[LULCConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize the negative sampler.

        Parameters
        ----------
        duckdb_connection : duckdb.DuckDBPyConnection
            Connection to DuckDB with geo_embeddings table
        lulc_config : LULCConfig, optional
            Configuration for LULC data source
        sampling_config : SamplingConfig, optional
            Configuration for sampling parameters
        """
        self.conn = duckdb_connection
        self.lulc_config = lulc_config or LULCConfig()
        self.sampling_config = sampling_config or SamplingConfig()
        self._ee_initialized = False

    def _init_earth_engine(self) -> None:
        """Initialize Earth Engine if not already done."""
        if not HAS_EE:
            raise ImportError(
                "Earth Engine (ee) and geemap required for LULC labeling. "
                "Install with: pip install earthengine-api geemap"
            )
        if not self._ee_initialized:
            try:
                ee.Initialize()
            except Exception:
                ee.Authenticate()
                ee.Initialize()
            self._ee_initialized = True

    def get_tile_centroids(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        limit: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Get tile centroids from DuckDB.

        Parameters
        ----------
        bounds : tuple, optional
            Bounding box (minx, miny, maxx, maxy) to filter tiles
        limit : int, optional
            Maximum number of tiles to return

        Returns
        -------
        GeoDataFrame with id, tile_id, and point geometry
        """
        self.conn.execute("LOAD spatial;")

        query = """
            SELECT id, tile_id, ST_AsText(geometry) as geom_wkt
            FROM geo_embeddings
            WHERE geometry IS NOT NULL
        """

        if bounds:
            minx, miny, maxx, maxy = bounds
            query += f"""
                AND ST_X(geometry) BETWEEN {minx} AND {maxx}
                AND ST_Y(geometry) BETWEEN {miny} AND {maxy}
            """

        if limit:
            query += f" LIMIT {limit}"

        df = self.conn.execute(query).fetchdf()

        # Convert WKT to geometry
        df["geometry"] = gpd.GeoSeries.from_wkt(df["geom_wkt"])
        gdf = gpd.GeoDataFrame(
            df.drop(columns=["geom_wkt"]), geometry="geometry", crs="EPSG:4326"
        )

        return gdf

    def label_with_lulc(
        self,
        points_gdf: gpd.GeoDataFrame,
        batch_size: int = 5000,
    ) -> gpd.GeoDataFrame:
        """
        Label points with ESRI LULC class using Earth Engine.

        Parameters
        ----------
        points_gdf : GeoDataFrame
            Points to label (must have geometry column)
        batch_size : int
            Number of points to process per EE request

        Returns
        -------
        GeoDataFrame with added 'lulc_class' and 'class' columns
        """
        self._init_earth_engine()

        # Load LULC image
        lulc_collection = ee.ImageCollection(self.lulc_config.collection)
        lulc_image = lulc_collection.filterDate(
            self.lulc_config.start_date, self.lulc_config.end_date
        ).mosaic()

        all_results = []

        # Process in batches to avoid EE limits
        for i in range(0, len(points_gdf), batch_size):
            batch = points_gdf.iloc[i : i + batch_size].copy()
            logging.info(f"Labeling batch {i//batch_size + 1}, {len(batch)} points")

            # Convert to EE FeatureCollection
            features = []
            for idx, row in batch.iterrows():
                point = ee.Geometry.Point([row.geometry.x, row.geometry.y])
                features.append(ee.Feature(point, {"id": int(row["id"])}))

            fc = ee.FeatureCollection(features)

            # Sample LULC values at points
            sampled = lulc_image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=10,
            )

            # Convert back to GeoDataFrame
            try:
                result_gdf = geemap.ee_to_gdf(sampled)
                result_gdf = result_gdf.rename(columns={"first": "lulc_class"})
                all_results.append(result_gdf)
            except Exception as e:
                logging.warning(f"Batch {i//batch_size + 1} failed: {e}")
                continue

        if not all_results:
            raise RuntimeError("All LULC labeling batches failed")

        # Combine results
        labeled_gdf = pd.concat(all_results, ignore_index=True)

        # Map class IDs to names
        labeled_gdf["class"] = labeled_gdf["lulc_class"].map(
            self.lulc_config.class_mapping
        )

        # Merge back with original data
        result = points_gdf.merge(
            labeled_gdf[["id", "lulc_class", "class"]], on="id", how="left"
        )

        return result

    def filter_near_positives(
        self,
        candidates_gdf: gpd.GeoDataFrame,
        positives_gdf: gpd.GeoDataFrame,
        buffer_meters: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """
        Remove candidate negatives that are too close to positive points.

        Parameters
        ----------
        candidates_gdf : GeoDataFrame
            Candidate negative points
        positives_gdf : GeoDataFrame
            Positive points to buffer around
        buffer_meters : float, optional
            Buffer distance in meters (default from sampling_config)

        Returns
        -------
        GeoDataFrame with filtered candidates
        """
        buffer_m = buffer_meters or self.sampling_config.buffer_meters

        if len(positives_gdf) == 0:
            return candidates_gdf

        # Determine UTM zone from first positive point
        first_point = positives_gdf.geometry.iloc[0]
        utm_zone = int(((first_point.x + 180) / 6) + 1)
        hemisphere = "N" if first_point.y >= 0 else "S"
        utm_epsg = (
            f"EPSG:326{utm_zone:02d}"
            if hemisphere == "N"
            else f"EPSG:327{utm_zone:02d}"
        )

        # Project to UTM for accurate buffering
        positives_utm = positives_gdf.to_crs(utm_epsg)
        candidates_utm = candidates_gdf.to_crs(utm_epsg)

        # Create buffer union around positives
        positive_buffer = positives_utm.geometry.buffer(buffer_m).unary_union

        # Filter out candidates that intersect buffer
        mask = ~candidates_utm.geometry.intersects(positive_buffer)
        filtered = candidates_gdf[mask].copy()

        logging.info(
            f"Filtered {len(candidates_gdf) - len(filtered)} candidates "
            f"within {buffer_m}m of positives"
        )

        return filtered

    def sample_stratified(
        self,
        labeled_gdf: gpd.GeoDataFrame,
        samples_per_class: Optional[Dict[str, int]] = None,
        random_seed: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Sample points stratified by LULC class.

        Parameters
        ----------
        labeled_gdf : GeoDataFrame
            Points with 'class' column
        samples_per_class : dict, optional
            Number of samples per class (default from sampling_config)
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        GeoDataFrame with stratified sample
        """
        samples_per_class = samples_per_class or self.sampling_config.samples_per_class
        seed = random_seed or self.sampling_config.random_seed

        np.random.seed(seed)

        sampled_dfs = []

        for class_name, n_samples in samples_per_class.items():
            class_points = labeled_gdf[labeled_gdf["class"] == class_name]

            if len(class_points) == 0:
                logging.warning(f"No points found for class '{class_name}'")
                continue

            n_to_sample = min(n_samples, len(class_points))
            sampled = class_points.sample(n=n_to_sample, random_state=seed)
            sampled_dfs.append(sampled)

            logging.info(f"Sampled {n_to_sample} points for class '{class_name}'")

        if not sampled_dfs:
            raise ValueError("No samples collected from any class")

        return gpd.GeoDataFrame(
            pd.concat(sampled_dfs, ignore_index=True), crs="EPSG:4326"
        )

    def sample_negatives(
        self,
        positives_gdf: gpd.GeoDataFrame,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        use_lulc: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Full pipeline to sample negative training examples.

        Parameters
        ----------
        positives_gdf : GeoDataFrame
            Positive training examples (to avoid)
        bounds : tuple, optional
            Bounding box to restrict sampling area
        use_lulc : bool
            Whether to use Earth Engine LULC for stratification

        Returns
        -------
        GeoDataFrame with sampled negatives including:
            - id: DuckDB embedding ID
            - tile_id: Tile identifier
            - geometry: Point geometry
            - class: LULC class name (if use_lulc=True)
            - label: 0 (negative)
        """
        logging.info("Getting tile centroids from DuckDB...")

        # Determine bounds from positives if not provided
        if bounds is None and len(positives_gdf) > 0:
            pos_bounds = positives_gdf.total_bounds
            # Expand bounds by 10%
            dx = (pos_bounds[2] - pos_bounds[0]) * 0.1
            dy = (pos_bounds[3] - pos_bounds[1]) * 0.1
            bounds = (
                pos_bounds[0] - dx,
                pos_bounds[1] - dy,
                pos_bounds[2] + dx,
                pos_bounds[3] + dy,
            )

        # Get candidate points from DuckDB
        candidates = self.get_tile_centroids(bounds=bounds)
        logging.info(f"Found {len(candidates)} candidate tiles")

        # Filter out points too close to positives
        candidates = self.filter_near_positives(candidates, positives_gdf)

        if use_lulc:
            # Label with LULC classes
            logging.info("Labeling candidates with LULC classes...")
            candidates = self.label_with_lulc(candidates)

            # Stratified sampling
            logging.info("Performing stratified sampling...")
            negatives = self.sample_stratified(candidates)
        else:
            # Random sampling without stratification
            n_samples = sum(self.sampling_config.samples_per_class.values())
            n_to_sample = min(n_samples, len(candidates))
            negatives = candidates.sample(
                n=n_to_sample, random_state=self.sampling_config.random_seed
            )
            negatives["class"] = "unknown"

        # Add label column
        negatives["label"] = 0

        logging.info(f"Sampled {len(negatives)} negative examples")

        return negatives


def sample_negatives_from_db(
    duckdb_path: str,
    positives_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    use_lulc: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convenience function to sample negatives from command line.

    Parameters
    ----------
    duckdb_path : str
        Path to DuckDB database
    positives_path : str
        Path to GeoJSON/Parquet with positive points
    output_path : str
        Path to save output GeoJSON
    config_path : str, optional
        Path to JSON config file
    use_lulc : bool
        Whether to use Earth Engine LULC stratification

    Returns
    -------
    GeoDataFrame with sampled negatives
    """
    import duckdb

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config if provided
    lulc_config = LULCConfig()
    sampling_config = SamplingConfig()

    if config_path:
        with open(config_path) as f:
            config = json.load(f)
        if "lulc" in config:
            lulc_config = LULCConfig(**config["lulc"])
        if "sampling" in config:
            sampling_config = SamplingConfig(**config["sampling"])

    # Load positives
    if positives_path.endswith(".parquet"):
        positives = gpd.read_parquet(positives_path)
    else:
        positives = gpd.read_file(positives_path)

    # Connect to DuckDB
    conn = duckdb.connect(duckdb_path, read_only=True)

    # Sample negatives
    sampler = NegativeSampler(conn, lulc_config, sampling_config)
    negatives = sampler.sample_negatives(positives, use_lulc=use_lulc)

    conn.close()

    # Save output
    negatives.to_file(output_path, driver="GeoJSON")
    logging.info(f"Saved {len(negatives)} negatives to {output_path}")

    return negatives


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample negatives from DuckDB embeddings"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument(
        "--positives", required=True, help="Path to positive points file"
    )
    parser.add_argument("--output", required=True, help="Output GeoJSON path")
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument(
        "--no-lulc", action="store_true", help="Skip LULC stratification"
    )

    args = parser.parse_args()

    sample_negatives_from_db(
        duckdb_path=args.db,
        positives_path=args.positives,
        output_path=args.output,
        config_path=args.config,
        use_lulc=not args.no_lulc,
    )
