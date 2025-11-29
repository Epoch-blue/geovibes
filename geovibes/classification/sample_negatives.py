"""
Sample negatives from ESRI LULC using Earth Engine stratified sampling.

Adapted from ei-notebook/src/sample_negatives.py

This module samples negative training examples directly from ESRI Global LULC
using Earth Engine's stratifiedSample(), filters points near positives,
then matches to DuckDB tiles to get tile_ids for the classification pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import duckdb
import geopandas as gpd
import pandas as pd
import shapely

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

        Uses UTM projection for accurate distance calculation in meters.

        Parameters
        ----------
        samples_gdf : GeoDataFrame
            Sampled points with geometry

        Returns
        -------
        GeoDataFrame with tile_id column added, filtered to only matched points
        """
        # Determine UTM zone from centroid of points
        center_lon = samples_gdf.geometry.x.mean()
        center_lat = samples_gdf.geometry.y.mean()
        utm_zone = int(((center_lon + 180) / 6) + 1)
        hemisphere = "N" if center_lat >= 0 else "S"
        utm_epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

        logging.info(
            f"Spatial matching {len(samples_gdf)} points using UTM zone {utm_zone}{hemisphere}..."
        )

        conn = duckdb.connect(self.duckdb_path, read_only=True)
        conn.execute("LOAD spatial;")

        results = []
        start_time = time.perf_counter()

        for idx, row in samples_gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y

            # Query finds nearest tile by distance in meters (UTM projection)
            query = f"""
                SELECT
                    tile_id,
                    ST_Distance(
                        ST_Transform(geometry, 'EPSG:4326', 'EPSG:{utm_epsg}'),
                        ST_Transform(ST_Point({lon}, {lat}), 'EPSG:4326', 'EPSG:{utm_epsg}')
                    ) as distance_m
                FROM geo_embeddings
                WHERE geometry IS NOT NULL
                ORDER BY distance_m
                LIMIT 1
            """

            result = conn.execute(query).fetchone()
            if result is None:
                logging.warning(f"No nearby tile found for point ({lon}, {lat})")
                continue

            tile_id, distance_m = result
            results.append({"idx": idx, "tile_id": tile_id, "distance_m": distance_m})

        conn.close()

        elapsed = time.perf_counter() - start_time
        logging.info(f"Spatial matching completed in {elapsed:.1f}s")

        # Add tile_id to dataframe
        match_df = pd.DataFrame(results).set_index("idx")
        samples_gdf = samples_gdf.copy()
        samples_gdf["tile_id"] = samples_gdf.index.map(match_df["tile_id"])
        samples_gdf["match_distance_m"] = samples_gdf.index.map(match_df["distance_m"])

        # Filter out unmatched points
        samples_gdf = samples_gdf.dropna(subset=["tile_id"])

        # Report statistics
        logging.info(f"  Matched {len(samples_gdf)} points")
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


def sample_negatives_cli(
    positives_path: str,
    aoi_path: str,
    duckdb_path: str,
    output_path: str,
    buffer_meters: float = 500.0,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Sample negatives from command line.

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
    parser.add_argument(
        "--positives", required=True, help="Path to positive points file"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI file (GeoJSON)")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--buffer", type=float, default=500.0, help="Buffer distance in meters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    sample_negatives_cli(
        positives_path=args.positives,
        aoi_path=args.aoi,
        duckdb_path=args.db,
        output_path=args.output,
        buffer_meters=args.buffer,
        seed=args.seed,
    )
