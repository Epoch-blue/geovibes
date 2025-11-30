"""Output generation for classification results."""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import pyproj
import time
import os


@dataclass
class OutputTiming:
    """Timing information for output generation."""

    fetch_metadata_sec: float
    generate_tiles_sec: float
    union_tiles_sec: float
    export_sec: float
    total_sec: float


class OutputGenerator:
    """Generates GeoJSON output from classification detections."""

    def __init__(
        self,
        duckdb_connection,
        tile_size_px: int = 32,
        tile_overlap_px: int = 16,
        resolution_m: float = 10.0,
    ):
        """
        Initialize output generator.

        Parameters
        ----------
        duckdb_connection : duckdb.DuckDBPyConnection
            DuckDB connection with geo_embeddings table
        tile_size_px : int
            Tile size in pixels
        tile_overlap_px : int
            Tile overlap in pixels
        resolution_m : float
            Resolution in meters per pixel
        """
        self.conn = duckdb_connection
        self.tile_size_m = tile_size_px * resolution_m
        self.half_tile_m = self.tile_size_m / 2

    def fetch_detection_metadata(
        self,
        detections: List[Tuple[int, float]],
        chunk_size: int = 10_000,
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """
        Fetch metadata for detected IDs.

        Parameters
        ----------
        detections : List[Tuple[int, float]]
            List of (id, probability) tuples
        chunk_size : int
            Number of IDs per query chunk

        Returns
        -------
        Tuple[gpd.GeoDataFrame, float]
            GeoDataFrame with point geometries and probability, time in seconds
        """
        start = time.perf_counter()

        detections_df = pd.DataFrame(detections, columns=["id", "probability"])

        all_results = []
        for i in range(0, len(detections_df), chunk_size):
            chunk = detections_df.iloc[i : i + chunk_size]
            ids = chunk["id"].tolist()

            ids_str = ",".join(str(id_val) for id_val in ids)
            query = f"""
                SELECT
                    id,
                    tile_id,
                    ST_AsText(geometry) as geometry_wkt
                FROM geo_embeddings
                WHERE id IN ({ids_str})
            """

            result = self.conn.execute(query).fetchdf()
            all_results.append(result)

        metadata_df = pd.concat(all_results, ignore_index=True)
        metadata_df = metadata_df.merge(detections_df, on="id", how="inner")
        metadata_df["geometry"] = gpd.GeoSeries.from_wkt(metadata_df["geometry_wkt"])
        metadata_df = metadata_df.drop(columns=["geometry_wkt"])

        gdf = gpd.GeoDataFrame(
            metadata_df,
            geometry="geometry",
            crs="EPSG:4326",
        )

        elapsed = time.perf_counter() - start
        return gdf, elapsed

    def generate_tile_geometries(
        self,
        detections_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """
        Convert point centroids to square tile polygons.

        Parameters
        ----------
        detections_gdf : gpd.GeoDataFrame
            GeoDataFrame with point geometries

        Returns
        -------
        Tuple[gpd.GeoDataFrame, float]
            GeoDataFrame with polygon geometries, time in seconds
        """
        start = time.perf_counter()

        detections_gdf["utm_crs"] = detections_gdf["tile_id"].apply(self._get_utm_crs)

        all_tiles = []
        for utm_crs, group in detections_gdf.groupby("utm_crs"):
            group_utm = group.to_crs(utm_crs)

            tiles = []
            for idx, row in group_utm.iterrows():
                point = row.geometry
                x, y = point.x, point.y
                square = box(
                    x - self.half_tile_m,
                    y - self.half_tile_m,
                    x + self.half_tile_m,
                    y + self.half_tile_m,
                )
                tiles.append(
                    {
                        "id": row["id"],
                        "tile_id": row["tile_id"],
                        "probability": row["probability"],
                        "geometry": square,
                    }
                )

            tiles_gdf = gpd.GeoDataFrame(tiles, crs=utm_crs)
            tiles_gdf = tiles_gdf.to_crs("EPSG:4326")
            all_tiles.append(tiles_gdf)

        result_gdf = gpd.GeoDataFrame(
            pd.concat(all_tiles, ignore_index=True), crs="EPSG:4326"
        )

        elapsed = time.perf_counter() - start
        return result_gdf, elapsed

    def union_tiles(
        self,
        tiles_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """
        Union overlapping tiles into larger polygons.

        Parameters
        ----------
        tiles_gdf : gpd.GeoDataFrame
            GeoDataFrame with tile polygons

        Returns
        -------
        Tuple[gpd.GeoDataFrame, float]
            GeoDataFrame with unioned geometries, time in seconds
        """
        start = time.perf_counter()

        union_geom = unary_union(tiles_gdf.geometry)

        if union_geom.geom_type == "Polygon":
            geometries = [union_geom]
        else:
            geometries = list(union_geom.geoms)

        union_gdf = gpd.GeoDataFrame(
            {"geometry": geometries}, geometry="geometry", crs="EPSG:4326"
        )

        elapsed = time.perf_counter() - start
        return union_gdf, elapsed

    def export_geojson(
        self,
        tiles_gdf: gpd.GeoDataFrame,
        union_gdf: gpd.GeoDataFrame,
        output_dir: str,
        name: str = "classification",
    ) -> Tuple[Dict[str, str], float]:
        """
        Export detection results to GeoJSON files.

        Parameters
        ----------
        tiles_gdf : gpd.GeoDataFrame
            Individual tile polygons with probabilities
        union_gdf : gpd.GeoDataFrame
            Unioned detection regions
        output_dir : str
            Output directory path
        name : str
            Output file name prefix

        Returns
        -------
        Tuple[Dict[str, str], float]
            Dict with output paths, time in seconds
        """
        start = time.perf_counter()

        os.makedirs(output_dir, exist_ok=True)

        detections_path = os.path.join(output_dir, f"{name}_detections.geojson")
        union_path = os.path.join(output_dir, f"{name}_union.geojson")

        tiles_gdf.to_file(detections_path, driver="GeoJSON")
        union_gdf.to_file(union_path, driver="GeoJSON")

        elapsed = time.perf_counter() - start

        return {
            "detections": detections_path,
            "union": union_path,
        }, elapsed

    def _get_utm_crs(self, tile_id: str) -> pyproj.CRS:
        """Parse MGRS tile_id to get UTM CRS."""
        zone_str = tile_id[:2]
        band = tile_id[2]
        zone = int(zone_str)

        if band >= "N":
            epsg_code = 32600 + zone
        else:
            epsg_code = 32700 + zone

        return pyproj.CRS.from_epsg(epsg_code)

    def generate_output(
        self,
        detections: List[Tuple[int, float]],
        output_dir: str,
        name: str = "classification",
        chunk_size: int = 10_000,
    ) -> Tuple[Dict[str, str], OutputTiming]:
        """
        Complete pipeline: fetch metadata, generate tiles, union, and export.

        Parameters
        ----------
        detections : List[Tuple[int, float]]
            List of (id, probability) tuples
        output_dir : str
            Output directory path
        name : str
            Output file name prefix
        chunk_size : int
            Number of IDs per metadata query chunk

        Returns
        -------
        Tuple[Dict[str, str], OutputTiming]
            Output paths dict and timing info
        """
        total_start = time.perf_counter()

        detections_gdf, fetch_time = self.fetch_detection_metadata(
            detections, chunk_size
        )

        tiles_gdf, generate_time = self.generate_tile_geometries(detections_gdf)

        union_gdf, union_time = self.union_tiles(tiles_gdf)

        output_paths, export_time = self.export_geojson(
            tiles_gdf, union_gdf, output_dir, name
        )

        total_time = time.perf_counter() - total_start

        timing = OutputTiming(
            fetch_metadata_sec=fetch_time,
            generate_tiles_sec=generate_time,
            union_tiles_sec=union_time,
            export_sec=export_time,
            total_sec=total_time,
        )

        return output_paths, timing
