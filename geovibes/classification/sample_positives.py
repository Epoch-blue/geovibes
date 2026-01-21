"""
Sample spatially distributed positive examples from a GeoJSON file.

This module provides functionality to:
1. Read a GeoJSON file with known positive locations (e.g., palm oil mills)
2. Sample points maximizing spatial distribution using k-means clustering
3. Match points to nearest tiles in DuckDB
4. Fetch embeddings and output in labeled dataset format
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import duckdb
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class PositiveSamplingConfig:
    """Configuration for positive sampling."""

    num_samples: int = 50
    seed: int = 42
    label_class: str = "palm_oil_mill"
    source: str = "truth"
    max_match_distance_m: float = 500.0
    filter_to_db_bounds: bool = True


class PositiveSampler:
    """
    Sample spatially distributed positive examples from a GeoJSON file.

    Uses k-means clustering to select points that maximize spatial coverage,
    then matches to DuckDB tiles and fetches embeddings.
    """

    def __init__(
        self,
        duckdb_path: str,
        config: Optional[PositiveSamplingConfig] = None,
    ):
        self.duckdb_path = duckdb_path
        self.config = config or PositiveSamplingConfig()
        self._id_column: Optional[str] = None

    def _get_id_column(self, conn: duckdb.DuckDBPyConnection) -> str:
        """Detect which ID column exists in the geo_embeddings table."""
        if self._id_column is not None:
            return self._id_column

        result = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'geo_embeddings' AND column_name IN ('tile_id', 'id')"
        ).fetchall()
        columns = {row[0] for row in result}
        self._id_column = "tile_id" if "tile_id" in columns else "id"
        return self._id_column

    def get_db_bounds(self) -> tuple:
        """Get spatial bounds of the database."""
        conn = duckdb.connect(self.duckdb_path, read_only=True)
        result = conn.execute(
            "SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM geo_embeddings"
        ).fetchone()
        conn.close()
        return result  # (min_lon, max_lon, min_lat, max_lat)

    def filter_to_db_bounds(
        self,
        gdf: gpd.GeoDataFrame,
        buffer_deg: float = 0.02,
    ) -> gpd.GeoDataFrame:
        """Filter points to those within database bounds."""
        min_lon, max_lon, min_lat, max_lat = self.get_db_bounds()

        mask = (
            (gdf.geometry.x >= min_lon - buffer_deg)
            & (gdf.geometry.x <= max_lon + buffer_deg)
            & (gdf.geometry.y >= min_lat - buffer_deg)
            & (gdf.geometry.y <= max_lat + buffer_deg)
        )

        filtered = gdf[mask].copy()
        logging.info(
            f"Filtered to {len(filtered)} of {len(gdf)} points within database bounds "
            f"[{min_lon:.2f}, {max_lon:.2f}] x [{min_lat:.2f}, {max_lat:.2f}]"
        )
        return filtered

    def sample_spatially_distributed(
        self,
        gdf: gpd.GeoDataFrame,
        num_samples: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Sample points maximizing spatial distribution using k-means clustering.

        Parameters
        ----------
        gdf : GeoDataFrame
            Input points to sample from
        num_samples : int, optional
            Number of samples (default from config)

        Returns
        -------
        GeoDataFrame with sampled points
        """
        num_samples = num_samples or self.config.num_samples

        if len(gdf) <= num_samples:
            logging.info(
                f"Input has {len(gdf)} points, less than requested {num_samples}. "
                "Returning all points."
            )
            return gdf.copy()

        coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

        kmeans = KMeans(
            n_clusters=num_samples,
            random_state=self.config.seed,
            n_init=10,
        )
        kmeans.fit(coords)

        selected_indices = []
        for cluster_idx in range(num_samples):
            cluster_mask = kmeans.labels_ == cluster_idx
            cluster_points = coords[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            centroid = kmeans.cluster_centers_[cluster_idx]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

        sampled = gdf.iloc[selected_indices].copy().reset_index(drop=True)
        logging.info(f"Selected {len(sampled)} spatially distributed points")

        return sampled

    def match_to_tiles(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Match points to nearest tiles in DuckDB using batch queries.

        Uses R-tree spatial index via ST_Intersects for efficient matching.

        Parameters
        ----------
        gdf : GeoDataFrame
            Points to match

        Returns
        -------
        GeoDataFrame with tile_id and match_distance_m columns added
        """
        conn = duckdb.connect(self.duckdb_path, read_only=True)
        conn.execute("LOAD spatial;")

        id_col = self._get_id_column(conn)

        gdf = gdf.copy()
        gdf["_idx"] = range(len(gdf))
        gdf["_lon"] = gdf.geometry.x
        gdf["_lat"] = gdf.geometry.y

        gdf["_utm_zone"] = ((gdf["_lon"] + 180) / 6).astype(int) + 1
        gdf["_hemisphere"] = np.where(gdf["_lat"] >= 0, "N", "S")

        zone_groups = list(gdf.groupby(["_utm_zone", "_hemisphere"]))
        logging.info(f"Matching {len(gdf)} points across {len(zone_groups)} UTM zone(s)")

        all_results = []
        for (zone, hemi), group in zone_groups:
            results = self._match_batch(conn, group, id_col)
            all_results.extend(results)
            logging.info(f"  Zone {zone}{hemi}: matched {len(results)} of {len(group)} points")

        conn.close()

        if not all_results:
            logging.error("No points matched to tiles!")
            gdf["tile_id"] = None
            gdf["match_distance_m"] = None
            return gdf.drop(columns=["_idx", "_lon", "_lat", "_utm_zone", "_hemisphere"])

        match_df = {r["idx"]: r for r in all_results}
        gdf["tile_id"] = gdf["_idx"].map(lambda i: match_df.get(i, {}).get("tile_id"))
        gdf["match_distance_m"] = gdf["_idx"].map(
            lambda i: match_df.get(i, {}).get("distance_m")
        )

        gdf = gdf.drop(columns=["_idx", "_lon", "_lat", "_utm_zone", "_hemisphere"])
        matched = gdf.dropna(subset=["tile_id"]).copy()
        logging.info(f"Matched {len(matched)} of {len(gdf)} points to tiles")

        far_matches = matched[
            matched["match_distance_m"] > self.config.max_match_distance_m
        ]
        if len(far_matches) > 0:
            logging.warning(
                f"{len(far_matches)} points matched >{self.config.max_match_distance_m}m away"
            )

        return matched

    def _match_batch(
        self,
        conn: duckdb.DuckDBPyConnection,
        group: gpd.GeoDataFrame,
        id_col: str,
    ) -> list:
        """Match a batch of points in a single UTM zone."""
        results = []
        buffer_deg = 0.005  # ~500m

        values_parts = []
        for _, row in group.iterrows():
            values_parts.append(f"({row['_idx']}, {row['_lon']}, {row['_lat']})")

        if not values_parts:
            return results

        values_clause = ", ".join(values_parts)

        query = f"""
            WITH sample_points AS (
                SELECT * FROM (VALUES {values_clause}) AS t(idx, lon, lat)
            ),
            candidates AS (
                SELECT
                    p.idx,
                    g.{id_col} as tile_id,
                    ST_Distance(g.geometry, ST_Point(p.lon, p.lat)) * 111000 as distance_m
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
                results.append({"idx": idx, "tile_id": tile_id, "distance_m": distance_m})

            matched_idxs = {r[0] for r in batch_results}
            unmatched = [row for _, row in group.iterrows() if row["_idx"] not in matched_idxs]

            if unmatched:
                for row in unmatched:
                    retry_query = f"""
                        SELECT
                            {id_col} as tile_id,
                            ST_Distance(geometry, ST_Point({row['_lon']}, {row['_lat']})) * 111000 as distance_m
                        FROM geo_embeddings
                        WHERE geometry IS NOT NULL
                          AND ST_Intersects(geometry, ST_Buffer(ST_Point({row['_lon']}, {row['_lat']}), 0.02))
                        ORDER BY distance_m
                        LIMIT 1
                    """
                    result = conn.execute(retry_query).fetchone()
                    if result:
                        results.append({
                            "idx": row["_idx"],
                            "tile_id": result[0],
                            "distance_m": result[1],
                        })
                    else:
                        logging.warning(
                            f"No tile found for point at ({row['_lon']:.4f}, {row['_lat']:.4f})"
                        )

        except Exception as e:
            logging.warning(f"Batch query failed: {e}, falling back to per-point")
            for _, row in group.iterrows():
                query = f"""
                    SELECT {id_col} as tile_id,
                        ST_Distance(geometry, ST_Point({row['_lon']}, {row['_lat']})) * 111000 as distance_m
                    FROM geo_embeddings
                    WHERE ST_Intersects(geometry, ST_Buffer(ST_Point({row['_lon']}, {row['_lat']}), 0.005))
                    ORDER BY distance_m
                    LIMIT 1
                """
                result = conn.execute(query).fetchone()
                if result:
                    results.append({
                        "idx": row["_idx"],
                        "tile_id": result[0],
                        "distance_m": result[1],
                    })

        return results

    def fetch_embeddings(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Fetch embeddings from DuckDB for matched points.

        Parameters
        ----------
        gdf : GeoDataFrame
            Points with tile_id column

        Returns
        -------
        GeoDataFrame with embedding column added
        """
        if "tile_id" not in gdf.columns or gdf["tile_id"].isna().all():
            raise ValueError("GeoDataFrame must have tile_id column with valid values")

        conn = duckdb.connect(self.duckdb_path, read_only=True)
        id_col = self._get_id_column(conn)

        tile_ids = gdf["tile_id"].tolist()
        tile_id_str = ",".join(str(tid) for tid in tile_ids)

        query = f"""
            SELECT {id_col} as tile_id, embedding
            FROM geo_embeddings
            WHERE {id_col} IN ({tile_id_str})
        """
        result = conn.execute(query).fetchall()
        conn.close()

        embedding_map = {row[0]: list(row[1]) for row in result}
        gdf["embedding"] = gdf["tile_id"].map(embedding_map)

        missing = gdf["embedding"].isna().sum()
        if missing > 0:
            logging.warning(f"{missing} points missing embeddings")

        return gdf

    def sample_positives(
        self,
        input_gdf: gpd.GeoDataFrame,
        num_samples: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Full pipeline to sample positive examples with embeddings.

        Parameters
        ----------
        input_gdf : GeoDataFrame
            Input points (known positive locations)
        num_samples : int, optional
            Number of samples

        Returns
        -------
        GeoDataFrame with sampled points including:
            - geometry: Point geometry
            - id: Tile ID from DuckDB
            - label: 1 (positive)
            - class: Label class name
            - embedding: Embedding vector
            - source: Source identifier
        """
        if self.config.filter_to_db_bounds:
            input_gdf = self.filter_to_db_bounds(input_gdf)
            if len(input_gdf) == 0:
                raise ValueError("No points within database bounds after filtering")

        sampled = self.sample_spatially_distributed(input_gdf, num_samples)
        matched = self.match_to_tiles(sampled)
        with_embeddings = self.fetch_embeddings(matched)

        tile_ids = with_embeddings["tile_id"].apply(
            lambda x: str(int(x)) if isinstance(x, float) else str(x)
        )
        output = gpd.GeoDataFrame(
            {
                "geometry": with_embeddings.geometry,
                "id": tile_ids,
                "label": 1,
                "class": self.config.label_class,
                "embedding": with_embeddings["embedding"],
                "source": self.config.source,
            },
            crs="EPSG:4326",
        )

        output = output.dropna(subset=["embedding"])
        logging.info(f"Final positive sample count: {len(output)}")

        return output


def to_labeled_geojson(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Save GeoDataFrame to labeled dataset GeoJSON format.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data with id, label, class, embedding, source columns
    output_path : str
        Output file path
    """
    features = []
    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.geometry.x, row.geometry.y],
            },
            "properties": {
                "id": str(row["id"]),
                "label": int(row["label"]),
                "class": row["class"],
                "embedding": row["embedding"],
                "source": row["source"],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    logging.info(f"Saved {len(features)} features to {output_path}")


def sample_positives_cli(
    input_path: str,
    duckdb_path: str,
    output_path: str,
    num_samples: int = 50,
    label_class: str = "palm_oil_mill",
    source: str = "truth",
    seed: int = 42,
    filter_to_db_bounds: bool = True,
) -> gpd.GeoDataFrame:
    """
    CLI interface for sampling positive examples.

    Parameters
    ----------
    input_path : str
        Path to input GeoJSON with positive locations
    duckdb_path : str
        Path to DuckDB database with geo_embeddings table
    output_path : str
        Path to save output GeoJSON
    num_samples : int
        Number of samples to select
    label_class : str
        Class name for the positive label
    source : str
        Source identifier
    seed : int
        Random seed for reproducibility
    filter_to_db_bounds : bool
        Whether to filter input points to database bounds

    Returns
    -------
    GeoDataFrame with sampled positives
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_gdf = gpd.read_file(input_path)
    logging.info(f"Loaded {len(input_gdf)} points from {input_path}")

    config = PositiveSamplingConfig(
        num_samples=num_samples,
        seed=seed,
        label_class=label_class,
        source=source,
        filter_to_db_bounds=filter_to_db_bounds,
    )

    sampler = PositiveSampler(duckdb_path=duckdb_path, config=config)
    result = sampler.sample_positives(input_gdf, num_samples)

    to_labeled_geojson(result, output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample spatially distributed positive examples from GeoJSON"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input GeoJSON with positive locations",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to DuckDB database with geo_embeddings table",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoJSON file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to select (default: 50)",
    )
    parser.add_argument(
        "--class",
        dest="label_class",
        default="palm_oil_mill",
        help="Class name for the positive label (default: palm_oil_mill)",
    )
    parser.add_argument(
        "--source",
        default="truth",
        help="Source identifier (default: truth)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-filter-bounds",
        action="store_true",
        help="Don't filter input points to database bounds (slower for large inputs)",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if not output_path.endswith(".geojson"):
        output_path = f"truth_dataset_{timestamp}.geojson"

    sample_positives_cli(
        input_path=args.input,
        duckdb_path=args.db,
        output_path=output_path,
        num_samples=args.num_samples,
        label_class=args.label_class,
        source=args.source,
        seed=args.seed,
        filter_to_db_bounds=not args.no_filter_bounds,
    )
