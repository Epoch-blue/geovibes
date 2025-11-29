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
    fetch_metadata_sec: float
    generate_tiles_sec: float
    union_tiles_sec: float
    export_sec: float
    total_sec: float


class OutputGenerator:
    def __init__(
        self,
        duckdb_connection,
        tile_size_px: int = 32,
        tile_overlap_px: int = 16,
        resolution_m: float = 10.0,
    ):
        self.conn = duckdb_connection
        self.tile_size_m = tile_size_px * resolution_m  # 320m
        self.half_tile_m = self.tile_size_m / 2  # 160m buffer

    def fetch_detection_metadata(
        self,
        detections: List[Tuple[int, float]],  # List of (id, probability)
        chunk_size: int = 10_000,
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """
        Fetch id, tile_id, geometry for detected IDs.
        Returns (GeoDataFrame with point geometries and probability, time_sec)
        """
        start = time.perf_counter()

        # Create DataFrame from detections
        detections_df = pd.DataFrame(detections, columns=["id", "probability"])

        # Split into chunks to avoid SQL query size limits
        all_results = []
        for i in range(0, len(detections_df), chunk_size):
            chunk = detections_df.iloc[i : i + chunk_size]
            ids = chunk["id"].tolist()

            # Build SQL query for this chunk
            ids_str = ",".join(str(id_val) for id_val in ids)
            query = f"""
                SELECT
                    id,
                    tile_id,
                    ST_AsText(geometry) as geometry_wkt
                FROM geo_embeddings
                WHERE id IN ({ids_str})
            """

            # Execute query and fetch results
            result = self.conn.execute(query).fetchdf()
            all_results.append(result)

        # Combine all chunks
        metadata_df = pd.concat(all_results, ignore_index=True)

        # Merge with probabilities
        metadata_df = metadata_df.merge(detections_df, on="id", how="inner")

        # Convert WKT to Shapely geometries
        metadata_df["geometry"] = gpd.GeoSeries.from_wkt(metadata_df["geometry_wkt"])
        metadata_df = metadata_df.drop(columns=["geometry_wkt"])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            metadata_df,
            geometry="geometry",
            crs="EPSG:4326",  # WGS84
        )

        elapsed = time.perf_counter() - start
        return gdf, elapsed

    def generate_tile_geometries(
        self,
        detections_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """
        Convert point centroids to square tile polygons.
        - Parse UTM zone from tile_id (e.g., "16SFD_32_16_10_0_0" -> zone 16N)
        - Group by UTM zone
        - Buffer points by half_tile_m with square cap
        - Reproject back to WGS84
        Returns (GeoDataFrame with polygon geometries, time_sec)
        """
        start = time.perf_counter()

        # Parse UTM CRS for each tile
        detections_gdf["utm_crs"] = detections_gdf["tile_id"].apply(self._get_utm_crs)

        # Group by UTM CRS
        all_tiles = []
        for utm_crs, group in detections_gdf.groupby("utm_crs"):
            # Reproject to UTM
            group_utm = group.to_crs(utm_crs)

            # Create square tiles by buffering points
            # Using cap_style=3 for square caps
            tiles = []
            for idx, row in group_utm.iterrows():
                point = row.geometry
                # Create square polygon by buffering with square cap
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

            # Create GeoDataFrame in UTM
            tiles_gdf = gpd.GeoDataFrame(tiles, crs=utm_crs)

            # Reproject back to WGS84
            tiles_gdf = tiles_gdf.to_crs("EPSG:4326")
            all_tiles.append(tiles_gdf)

        # Combine all tiles from different UTM zones
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
        Returns (GeoDataFrame with unioned geometries, time_sec)
        """
        start = time.perf_counter()

        # Union all geometries
        union_geom = unary_union(tiles_gdf.geometry)

        # Handle both single polygon and multipolygon results
        if union_geom.geom_type == "Polygon":
            geometries = [union_geom]
        else:  # MultiPolygon
            geometries = list(union_geom.geoms)

        # Create GeoDataFrame with unioned geometries
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
        Export two files:
        - {name}_detections.geojson: Individual tile polygons with probabilities
        - {name}_union.geojson: Unioned detection regions
        Returns (dict with paths, time_sec)
        """
        start = time.perf_counter()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define output paths
        detections_path = os.path.join(output_dir, f"{name}_detections.geojson")
        union_path = os.path.join(output_dir, f"{name}_union.geojson")

        # Export detections (individual tiles with probabilities)
        tiles_gdf.to_file(detections_path, driver="GeoJSON")

        # Export union (merged regions)
        union_gdf.to_file(union_path, driver="GeoJSON")

        elapsed = time.perf_counter() - start

        return {
            "detections": detections_path,
            "union": union_path,
        }, elapsed

    def _get_utm_crs(self, tile_id: str) -> pyproj.CRS:
        """Parse MGRS tile_id to get UTM CRS. E.g., '16SFD...' -> EPSG:32616 (UTM 16N)"""
        # MGRS format: {zone}{band}{square}_...
        # Extract zone (first 2 digits) and band (letter after zone)
        zone_str = tile_id[:2]
        band = tile_id[2]

        zone = int(zone_str)

        # Bands N-X are Northern hemisphere, C-M are Southern hemisphere
        # Northern: EPSG:326{zone}, Southern: EPSG:327{zone}
        if band >= "N":
            epsg_code = 32600 + zone  # UTM North
        else:
            epsg_code = 32700 + zone  # UTM South

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
        Returns (output paths dict, timing info)
        """
        total_start = time.perf_counter()

        # Step 1: Fetch metadata
        detections_gdf, fetch_time = self.fetch_detection_metadata(
            detections, chunk_size
        )

        # Step 2: Generate tile geometries
        tiles_gdf, generate_time = self.generate_tile_geometries(detections_gdf)

        # Step 3: Union overlapping tiles
        union_gdf, union_time = self.union_tiles(tiles_gdf)

        # Step 4: Export to GeoJSON
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
