"""
Sample negatives from ESRI LULC using Earth Engine stratified sampling.

Adapted from ei-notebook/src/sample_negatives.py

This module samples negative training examples directly from ESRI Global LULC
using Earth Engine's stratifiedSample(), filters points near positives,
then matches to DuckDB tiles to get tile_ids for the classification pipeline.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import duckdb
import geopandas as gpd
import pandas as pd
import shapely
import yaml

# Earth Engine imports - required for this module
import ee
import geemap


@dataclass
class LULCConfig:
    """Configuration for ESRI LULC data source."""

    collection: str = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    # Remap ESRI classes to simplified classes
    # ESRI classes: 1=water, 2=trees, 4=flooded_veg, 5=crops, 7=built, 8=bare, 9=snow, 10=clouds, 11=rangeland
    input_classes: list = field(default_factory=lambda: [1, 2, 4, 5, 7, 8, 9, 10, 11])
    output_classes: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9])
    class_names: Dict[int, str] = field(
        default_factory=lambda: {
            1: "water",
            2: "trees",
            3: "flooded_vegetation",
            4: "crops",
            5: "built_area",
            6: "bare_ground",
            7: "snow_ice",
            8: "clouds",
            9: "rangeland",
        }
    )


@dataclass
class SamplingConfig:
    """Configuration for negative sampling."""

    # Number of samples per remapped class
    samples_per_class: Dict[int, int] = field(
        default_factory=lambda: {
            2: 500,  # trees
            4: 500,  # crops
            5: 200,  # built_area
            6: 200,  # bare_ground
            9: 300,  # rangeland
        }
    )
    scale: int = 10  # Sampling scale in meters (ESRI LULC is 10m)
    buffer_meters: float = 500.0  # Min distance from positive points
    seed: int = 42


@dataclass
class NegativeSamplingConfig:
    """Complete configuration for negative sampling pipeline."""

    # Required path (in config)
    aoi_path: str

    # Paths passed via CLI (not in config)
    positives_path: str = ""
    duckdb_path: str = ""
    output_path: str = ""

    # Sampling parameters
    samples_per_class: Dict[int, int] = field(
        default_factory=lambda: {
            2: 500,  # trees
            4: 500,  # crops
            5: 200,  # built_area
            6: 200,  # bare_ground
            9: 300,  # rangeland
        }
    )
    scale: int = 10
    buffer_meters: float = 500.0
    seed: int = 42

    # LULC configuration
    lulc_collection: str = (
        "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
    )
    lulc_start_date: str = "2023-01-01"
    lulc_end_date: str = "2023-12-31"
    lulc_input_classes: list = field(
        default_factory=lambda: [1, 2, 4, 5, 7, 8, 9, 10, 11]
    )
    lulc_output_classes: list = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    lulc_class_names: Dict[int, str] = field(
        default_factory=lambda: {
            1: "water",
            2: "trees",
            3: "flooded_vegetation",
            4: "crops",
            5: "built_area",
            6: "bare_ground",
            7: "snow_ice",
            8: "clouds",
            9: "rangeland",
        }
    )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "NegativeSamplingConfig":
        """
        Load configuration from YAML or JSON file.

        Parameters
        ----------
        config_path : str or Path
            Path to config file (.yaml, .yml, or .json)

        Returns
        -------
        NegativeSamplingConfig
        """
        config_path = Path(config_path)

        with open(config_path) as f:
            if config_path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )

        # Convert string keys to int for samples_per_class and class_names
        if "samples_per_class" in data:
            data["samples_per_class"] = {
                int(k): v for k, v in data["samples_per_class"].items()
            }
        if "lulc_class_names" in data:
            data["lulc_class_names"] = {
                int(k): v for k, v in data["lulc_class_names"].items()
            }

        return cls(**data)

    def to_lulc_config(self) -> LULCConfig:
        """Convert to LULCConfig."""
        return LULCConfig(
            collection=self.lulc_collection,
            start_date=self.lulc_start_date,
            end_date=self.lulc_end_date,
            input_classes=self.lulc_input_classes,
            output_classes=self.lulc_output_classes,
            class_names=self.lulc_class_names,
        )

    def to_sampling_config(self) -> SamplingConfig:
        """Convert to SamplingConfig."""
        return SamplingConfig(
            samples_per_class=self.samples_per_class,
            scale=self.scale,
            buffer_meters=self.buffer_meters,
            seed=self.seed,
        )


class NegativeSampler:
    """
    Sample negative training examples using Earth Engine stratified sampling.

    Uses ESRI Global LULC to sample points stratified by landcover class,
    filters samples near positives, then matches to DuckDB for tile_ids.
    """

    def __init__(
        self,
        duckdb_path: str,
        lulc_config: Optional[LULCConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize the negative sampler.

        Parameters
        ----------
        duckdb_path : str
            Path to DuckDB database with geo_embeddings table
        lulc_config : LULCConfig, optional
            Configuration for LULC data source
        sampling_config : SamplingConfig, optional
            Configuration for sampling parameters
        """
        self.duckdb_path = duckdb_path
        self.lulc_config = lulc_config or LULCConfig()
        self.sampling_config = sampling_config or SamplingConfig()
        self._ee_initialized = False

    @classmethod
    def from_config(cls, config: NegativeSamplingConfig) -> "NegativeSampler":
        """
        Create NegativeSampler from a NegativeSamplingConfig.

        Parameters
        ----------
        config : NegativeSamplingConfig
            Complete configuration

        Returns
        -------
        NegativeSampler
        """
        return cls(
            duckdb_path=config.duckdb_path,
            lulc_config=config.to_lulc_config(),
            sampling_config=config.to_sampling_config(),
        )

    def _init_earth_engine(self) -> None:
        """Initialize Earth Engine if not already done."""
        if not self._ee_initialized:
            try:
                ee.Initialize()
            except Exception:
                ee.Authenticate()
                ee.Initialize()
            self._ee_initialized = True

    def _get_utm_crs(self, gdf: gpd.GeoDataFrame) -> str:
        """Determine UTM CRS from first point in GeoDataFrame."""
        first_point = gdf.geometry.iloc[0]
        if hasattr(first_point, "centroid"):
            first_point = first_point.centroid
        utm_zone = int(((first_point.x + 180) / 6) + 1)
        hemisphere = "N" if first_point.y >= 0 else "S"
        return (
            f"EPSG:326{utm_zone:02d}"
            if hemisphere == "N"
            else f"EPSG:327{utm_zone:02d}"
        )

    def filter_near_positives(
        self,
        samples_gdf: gpd.GeoDataFrame,
        positives_gdf: gpd.GeoDataFrame,
        buffer_meters: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """
        Remove samples that are too close to positive points.

        Parameters
        ----------
        samples_gdf : GeoDataFrame
            Sample points to filter
        positives_gdf : GeoDataFrame
            Positive points to buffer around
        buffer_meters : float, optional
            Buffer distance in meters (default from sampling_config)

        Returns
        -------
        GeoDataFrame with filtered samples in EPSG:4326
        """
        buffer_m = buffer_meters or self.sampling_config.buffer_meters

        if len(positives_gdf) == 0:
            return samples_gdf

        utm_crs = self._get_utm_crs(positives_gdf)

        # Project to UTM for accurate buffering
        positives_utm = positives_gdf.to_crs(utm_crs)
        samples_utm = samples_gdf.to_crs(utm_crs)

        # Create buffer union around positives
        positive_buffer = positives_utm.geometry.buffer(buffer_m).unary_union

        # Filter out samples that intersect buffer
        mask = ~samples_utm.geometry.intersects(positive_buffer)
        filtered = samples_gdf[mask].copy()

        logging.info(
            f"Filtered {len(samples_gdf) - len(filtered)} samples "
            f"within {buffer_m}m of positives"
        )

        return filtered

    def sample_from_lulc(
        self,
        aoi: gpd.GeoDataFrame,
        samples_per_class: Optional[Dict[int, int]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Sample points from LULC raster using Earth Engine stratifiedSample.

        Parameters
        ----------
        aoi : GeoDataFrame
            Area of interest (single polygon)
        samples_per_class : dict, optional
            Number of samples per remapped class (default from sampling_config)

        Returns
        -------
        GeoDataFrame with sampled points and 'remapped' class column
        """
        self._init_earth_engine()

        samples_per_class = samples_per_class or self.sampling_config.samples_per_class

        # Load and remap LULC
        lulc_collection = ee.ImageCollection(self.lulc_config.collection)
        lulc_image = lulc_collection.filterDate(
            self.lulc_config.start_date, self.lulc_config.end_date
        ).mosaic()
        lulc_remapped = lulc_image.remap(
            self.lulc_config.input_classes, self.lulc_config.output_classes
        )

        # Convert AOI to EE geometry
        aoi_geom = aoi.geometry.iloc[0]
        ee_region = ee.Geometry(shapely.geometry.mapping(aoi_geom))

        # Build class values and points lists
        class_values = list(samples_per_class.keys())
        class_points = [samples_per_class[c] for c in class_values]

        logging.info(f"Sampling from LULC classes: {class_values}")
        logging.info(f"Samples per class: {class_points}")

        # Stratified sampling from LULC raster
        samples = lulc_remapped.stratifiedSample(
            region=ee_region,
            scale=self.sampling_config.scale,
            numPoints=0,  # Use classPoints instead
            classValues=class_values,
            classPoints=class_points,
            seed=self.sampling_config.seed,
            geometries=True,
        )

        # Convert to GeoDataFrame
        samples_gdf = geemap.ee_to_gdf(samples)
        logging.info(f"Sampled {len(samples_gdf)} points from LULC")

        return samples_gdf

    def spatial_match_tiles(self, samples_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Match sampled points to nearest tiles in DuckDB.

        Optimizations:
        - Groups points by UTM zone for accurate distance calculation
        - Uses bounding box pre-filter to avoid full table scans
        - Processes UTM zones in parallel

        Parameters
        ----------
        samples_gdf : GeoDataFrame
            Sampled points with geometry

        Returns
        -------
        GeoDataFrame with tile_id column added, filtered to only matched points
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        start_time = time.perf_counter()

        # Prepare data with UTM zone info
        samples_gdf = samples_gdf.copy()
        samples_gdf["_idx"] = samples_gdf.index
        samples_gdf["_lon"] = samples_gdf.geometry.x
        samples_gdf["_lat"] = samples_gdf.geometry.y

        # Calculate UTM zone for each point
        samples_gdf["_utm_zone"] = ((samples_gdf["_lon"] + 180) / 6).astype(int) + 1
        samples_gdf["_hemisphere"] = np.where(samples_gdf["_lat"] >= 0, "N", "S")

        # Group by UTM zone
        zone_groups = list(samples_gdf.groupby(["_utm_zone", "_hemisphere"]))

        logging.info(
            f"Spatial matching {len(samples_gdf)} points across "
            f"{len(zone_groups)} UTM zone(s)..."
        )

        # Process zones in parallel
        all_results = []
        max_workers = min(4, len(zone_groups))

        if max_workers == 1:
            # Single zone - no need for threading overhead
            (zone, hemi), group = zone_groups[0]
            results = self._match_zone_batch(group, zone, hemi)
            all_results.extend(results)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._match_zone_batch, group, zone, hemi): (
                        zone,
                        hemi,
                    )
                    for (zone, hemi), group in zone_groups
                }
                for future in as_completed(futures):
                    zone, hemi = futures[future]
                    results = future.result()
                    all_results.extend(results)
                    logging.info(f"  Zone {zone}{hemi}: matched {len(results)} points")

        elapsed = time.perf_counter() - start_time
        logging.info(f"Spatial matching completed in {elapsed:.1f}s")

        # Merge results back to dataframe
        if not all_results:
            logging.warning("No points matched!")
            samples_gdf["tile_id"] = None
            samples_gdf["match_distance_m"] = None
            return samples_gdf.dropna(subset=["tile_id"])

        match_df = pd.DataFrame(all_results).set_index("idx")
        samples_gdf["tile_id"] = samples_gdf["_idx"].map(match_df["tile_id"])
        samples_gdf["match_distance_m"] = samples_gdf["_idx"].map(
            match_df["distance_m"]
        )

        # Clean up temp columns
        samples_gdf = samples_gdf.drop(
            columns=["_idx", "_lon", "_lat", "_utm_zone", "_hemisphere"]
        )

        # Filter out unmatched points
        samples_gdf = samples_gdf.dropna(subset=["tile_id"])

        # Report statistics
        logging.info(f"  Matched {len(samples_gdf)} points total")
        logging.info(
            f"  Mean match distance: {samples_gdf['match_distance_m'].mean():.1f}m"
        )
        logging.info(
            f"  Max match distance: {samples_gdf['match_distance_m'].max():.1f}m"
        )

        # Warn if any matches are far
        far_matches = samples_gdf[samples_gdf["match_distance_m"] > 500]
        if len(far_matches) > 0:
            logging.warning(f"  {len(far_matches)} points matched >500m away")

        return samples_gdf

    def _match_zone_batch(
        self, group_gdf: gpd.GeoDataFrame, zone: int, hemisphere: str
    ) -> list:
        """
        Match points in a single UTM zone using batch query with R-tree index.

        Uses ST_Intersects to leverage the spatial R-tree index (10x+ faster
        than ST_X/ST_Y BETWEEN which forces full table scans).

        Processes points in batches using a single query with DISTINCT ON
        for efficient nearest-neighbor lookup.

        Parameters
        ----------
        group_gdf : GeoDataFrame
            Points in this UTM zone
        zone : int
            UTM zone number
        hemisphere : str
            'N' or 'S'

        Returns
        -------
        List of dicts with idx, tile_id, distance_m
        """

        utm_epsg = 32600 + zone if hemisphere == "N" else 32700 + zone

        # Each thread gets its own connection
        conn = duckdb.connect(self.duckdb_path, read_only=True)
        conn.execute("LOAD spatial;")

        results = []

        # Process in batches for efficient querying
        batch_size = 50
        points_list = list(group_gdf.iterrows())

        for batch_start in range(0, len(points_list), batch_size):
            batch = points_list[batch_start : batch_start + batch_size]

            # Build VALUES clause for batch query
            values_parts = []
            for _, row in batch:
                idx = row["_idx"]
                lon, lat = row["_lon"], row["_lat"]
                values_parts.append(f"({idx}, {lon}, {lat})")

            if not values_parts:
                continue

            values_clause = ", ".join(values_parts)

            # Buffer in degrees (approximately 500m)
            # Using a fixed buffer since ST_Intersects with R-tree is fast
            buffer_deg = 0.005  # ~500m at mid-latitudes

            # Batch query using DISTINCT ON pattern - leverages R-tree index
            # via ST_Intersects instead of ST_X/ST_Y BETWEEN (which forces table scan)
            query = f"""
                WITH sample_points AS (
                    SELECT * FROM (VALUES {values_clause}) AS t(idx, lon, lat)
                ),
                candidates AS (
                    SELECT
                        p.idx,
                        g.tile_id,
                        ST_Distance(
                            ST_Transform(g.geometry, 'EPSG:4326', 'EPSG:{utm_epsg}'),
                            ST_Transform(ST_Point(p.lon, p.lat), 'EPSG:4326', 'EPSG:{utm_epsg}')
                        ) as distance_m
                    FROM sample_points p
                    JOIN geo_embeddings g ON ST_Intersects(
                        g.geometry,
                        ST_Buffer(ST_Point(p.lon, p.lat), {buffer_deg})
                    )
                    WHERE g.geometry IS NOT NULL
                )
                SELECT DISTINCT ON (idx) idx, tile_id, distance_m
                FROM candidates
                ORDER BY idx, distance_m
            """

            try:
                batch_results = conn.execute(query).fetchall()
                for idx, tile_id, distance_m in batch_results:
                    results.append(
                        {"idx": idx, "tile_id": tile_id, "distance_m": distance_m}
                    )

                # Check for unmatched points and retry with larger buffer
                matched_idxs = {r[0] for r in batch_results}
                unmatched = [
                    (_, row) for _, row in batch if row["_idx"] not in matched_idxs
                ]

                if unmatched:
                    # Retry unmatched points with larger buffer (2km)
                    for _, row in unmatched:
                        idx = row["_idx"]
                        lon, lat = row["_lon"], row["_lat"]

                        retry_query = f"""
                            SELECT
                                tile_id,
                                ST_Distance(
                                    ST_Transform(geometry, 'EPSG:4326', 'EPSG:{utm_epsg}'),
                                    ST_Transform(ST_Point({lon}, {lat}), 'EPSG:4326', 'EPSG:{utm_epsg}')
                                ) as distance_m
                            FROM geo_embeddings
                            WHERE geometry IS NOT NULL
                              AND ST_Intersects(
                                  geometry,
                                  ST_Buffer(ST_Point({lon}, {lat}), 0.02)
                              )
                            ORDER BY distance_m
                            LIMIT 1
                        """
                        result = conn.execute(retry_query).fetchone()
                        if result:
                            results.append(
                                {
                                    "idx": idx,
                                    "tile_id": result[0],
                                    "distance_m": result[1],
                                }
                            )
                        else:
                            logging.warning(
                                f"No tile found within 2km of ({lon:.4f}, {lat:.4f})"
                            )

            except Exception as e:
                logging.warning(f"Batch query failed: {e}, falling back to per-point")
                # Fallback to per-point queries
                for _, row in batch:
                    idx = row["_idx"]
                    lon, lat = row["_lon"], row["_lat"]

                    query = f"""
                        SELECT tile_id,
                            ST_Distance(
                                ST_Transform(geometry, 'EPSG:4326', 'EPSG:{utm_epsg}'),
                                ST_Transform(ST_Point({lon}, {lat}), 'EPSG:4326', 'EPSG:{utm_epsg}')
                            ) as distance_m
                        FROM geo_embeddings
                        WHERE ST_Intersects(
                            geometry,
                            ST_Buffer(ST_Point({lon}, {lat}), 0.005)
                        )
                        ORDER BY distance_m
                        LIMIT 1
                    """
                    result = conn.execute(query).fetchone()
                    if result:
                        results.append(
                            {"idx": idx, "tile_id": result[0], "distance_m": result[1]}
                        )

        conn.close()
        return results

    def sample_negatives(
        self,
        positives_gdf: gpd.GeoDataFrame,
        aoi: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Full pipeline to sample negative training examples.

        Parameters
        ----------
        positives_gdf : GeoDataFrame
            Positive training examples (to avoid)
        aoi : GeoDataFrame
            Area of interest (required)

        Returns
        -------
        GeoDataFrame with sampled negatives including:
            - geometry: Point geometry
            - tile_id: Matched tile ID from DuckDB
            - class: LULC class name
            - label: 0 (negative)
        """
        # Sample from LULC
        logging.info("Sampling negatives from ESRI LULC...")
        samples = self.sample_from_lulc(aoi)

        # Remove exact geometry matches with positives
        before = len(samples)
        samples = samples[~samples.geometry.isin(positives_gdf.geometry)]
        after = len(samples)
        if before > after:
            logging.info(f"Removed {before - after} exact geometry matches")

        # Filter near positives
        samples = self.filter_near_positives(samples, positives_gdf)

        # Spatial match to get tile_ids
        logging.info("Matching samples to DuckDB tiles...")
        samples = self.spatial_match_tiles(samples)

        # Map class integers to names
        samples["class"] = samples["remapped"].map(self.lulc_config.class_names)

        # Add label column
        samples["label"] = 0

        # Select final columns
        final_columns = ["geometry", "tile_id", "class", "label"]
        if "match_distance_m" in samples.columns:
            final_columns.append("match_distance_m")
        samples = samples[final_columns]

        logging.info(f"Final negative sample count: {len(samples)}")

        return samples


def run_from_config(
    config_path: Union[str, Path],
    positives_path: str,
    duckdb_path: str,
    output_path: str,
) -> gpd.GeoDataFrame:
    """
    Run negative sampling pipeline from a config file.

    Parameters
    ----------
    config_path : str or Path
        Path to config file (.yaml, .yml, or .json)
    positives_path : str
        Path to positive points file (GeoJSON or Parquet)
    duckdb_path : str
        Path to DuckDB database with geo_embeddings table
    output_path : str
        Path to save output (GeoJSON or Parquet)

    Returns
    -------
    GeoDataFrame with sampled negatives
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    config = NegativeSamplingConfig.from_file(config_path)
    # Override paths from CLI
    config.positives_path = positives_path
    config.duckdb_path = duckdb_path
    config.output_path = output_path
    logging.info(f"Loaded config from {config_path}")

    # Load positives
    if config.positives_path.endswith(".parquet"):
        positives = gpd.read_parquet(config.positives_path)
    else:
        positives = gpd.read_file(config.positives_path)
    logging.info(f"Loaded {len(positives)} positive points")

    # Load AOI
    aoi = gpd.read_file(config.aoi_path)
    logging.info(f"Loaded AOI from {config.aoi_path}")

    # Create sampler and run
    sampler = NegativeSampler.from_config(config)
    negatives = sampler.sample_negatives(positives, aoi=aoi)

    # Save output
    if config.output_path.endswith(".parquet"):
        negatives.to_parquet(config.output_path)
    else:
        negatives.to_file(config.output_path, driver="GeoJSON")
    logging.info(f"Saved {len(negatives)} negatives to {config.output_path}")

    return negatives


def sample_negatives_cli(
    positives_path: str,
    aoi_path: str,
    duckdb_path: str,
    output_path: str,
    buffer_meters: float = 500.0,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Sample negatives from command line arguments (legacy interface).

    Parameters
    ----------
    positives_path : str
        Path to positive points file (GeoJSON or Parquet)
    aoi_path : str
        Path to AOI file (GeoJSON)
    duckdb_path : str
        Path to DuckDB database with geo_embeddings table
    output_path : str
        Path to save output (GeoJSON or Parquet)
    buffer_meters : float
        Buffer distance from positives
    seed : int
        Random seed

    Returns
    -------
    GeoDataFrame with sampled negatives
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load positives
    if positives_path.endswith(".parquet"):
        positives = gpd.read_parquet(positives_path)
    else:
        positives = gpd.read_file(positives_path)

    # Load AOI
    aoi = gpd.read_file(aoi_path)

    # Sample negatives
    sampling_config = SamplingConfig(buffer_meters=buffer_meters, seed=seed)
    sampler = NegativeSampler(duckdb_path=duckdb_path, sampling_config=sampling_config)
    negatives = sampler.sample_negatives(positives, aoi=aoi)

    # Save output
    if output_path.endswith(".parquet"):
        negatives.to_parquet(output_path)
    else:
        negatives.to_file(output_path, driver="GeoJSON")
    logging.info(f"Saved {len(negatives)} negatives to {output_path}")

    return negatives


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample negatives from ESRI LULC using Earth Engine"
    )

    # Required CLI arguments
    parser.add_argument(
        "--positives", required=True, help="Path to positive points file"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output", required=True, help="Output file path")

    # Config file (contains AOI, sampling params, LULC config)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (.yaml or .json) with AOI and sampling parameters",
    )

    args = parser.parse_args()

    run_from_config(
        config_path=args.config,
        positives_path=args.positives,
        duckdb_path=args.db,
        output_path=args.output,
    )
