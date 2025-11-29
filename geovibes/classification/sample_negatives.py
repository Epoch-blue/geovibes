"""
Sample negatives from ESRI LULC using Earth Engine stratified sampling.

Adapted from ei-notebook/src/sample_negatives.py

This module samples negative training examples directly from ESRI Global LULC
using Earth Engine's stratifiedSample(), then filters out points too close
to positive examples.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import geopandas as gpd
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
    then filters out samples too close to positive examples.
    """

    def __init__(
        self,
        lulc_config: Optional[LULCConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize the negative sampler.

        Parameters
        ----------
        lulc_config : LULCConfig, optional
            Configuration for LULC data source
        sampling_config : SamplingConfig, optional
            Configuration for sampling parameters
        """
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

    def sample_negatives(
        self,
        positives_gdf: gpd.GeoDataFrame,
        aoi: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """
        Full pipeline to sample negative training examples.

        Parameters
        ----------
        positives_gdf : GeoDataFrame
            Positive training examples (to avoid)
        aoi : GeoDataFrame, optional
            Area of interest. If None, uses bounding box of positives expanded by 10%

        Returns
        -------
        GeoDataFrame with sampled negatives including:
            - geometry: Point geometry
            - class: LULC class name
            - label: 0 (negative)
        """
        # Determine AOI from positives if not provided
        if aoi is None:
            bounds = positives_gdf.total_bounds
            dx = (bounds[2] - bounds[0]) * 0.1
            dy = (bounds[3] - bounds[1]) * 0.1
            from shapely.geometry import box

            aoi_geom = box(
                bounds[0] - dx,
                bounds[1] - dy,
                bounds[2] + dx,
                bounds[3] + dy,
            )
            aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")

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

        # Map class integers to names
        samples["class"] = samples["remapped"].map(self.lulc_config.class_names)

        # Add label column
        samples["label"] = 0

        logging.info(f"Final negative sample count: {len(samples)}")

        return samples


def sample_negatives_cli(
    positives_path: str,
    output_path: str,
    aoi_path: Optional[str] = None,
    buffer_meters: float = 500.0,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Convenience function to sample negatives from command line.

    Parameters
    ----------
    positives_path : str
        Path to positive points file (GeoJSON or Parquet)
    output_path : str
        Path to save output (GeoJSON or Parquet)
    aoi_path : str, optional
        Path to AOI file
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

    # Load AOI if provided
    aoi = None
    if aoi_path:
        aoi = gpd.read_file(aoi_path)

    # Sample negatives
    sampling_config = SamplingConfig(buffer_meters=buffer_meters, seed=seed)
    sampler = NegativeSampler(sampling_config=sampling_config)
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
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--aoi", help="Optional AOI file path")
    parser.add_argument(
        "--buffer", type=float, default=500.0, help="Buffer distance in meters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    sample_negatives_cli(
        positives_path=args.positives,
        output_path=args.output,
        aoi_path=args.aoi,
        buffer_meters=args.buffer,
        seed=args.seed,
    )
