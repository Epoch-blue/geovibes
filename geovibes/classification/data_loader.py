"""
Data loader for classification training.

Loads GeoJSON training data and fetches corresponding embeddings from DuckDB.
Supports both tile_id matching and spatial (nearest point) matching.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import json
import time
import duckdb


@dataclass
class LoaderTiming:
    """Timing information for data loading operations."""

    parse_geojson_sec: float
    spatial_match_sec: float
    fetch_embeddings_sec: float
    stratified_split_sec: float
    total_sec: float


class ClassificationDataLoader:
    """
    Loads training data from GeoJSON and fetches embeddings from DuckDB.

    Supports two matching modes:
    1. tile_id matching: GeoJSON has tile_id property that matches DuckDB
    2. Spatial matching: GeoJSON has point geometry, finds nearest embedding

    Supports stratified train/test split with equal negatives per class.
    """

    def __init__(self, duckdb_connection: duckdb.DuckDBPyConnection, geojson_path: str):
        """
        Initialize the data loader.

        Args:
            duckdb_connection: Active DuckDB connection to database with geo_embeddings table
            geojson_path: Path to GeoJSON file with label, class properties
                         and either tile_id property OR point geometry
        """
        self.conn = duckdb_connection
        self.geojson_path = geojson_path

    def load(
        self, test_fraction: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, LoaderTiming]:
        """
        Load and split training data.

        Args:
            test_fraction: Fraction of data to use for test set (0.0 to 1.0)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df, timing) where DataFrames have columns:
                - tile_id: str
                - embedding: np.ndarray (float32)
                - label: int (0 or 1)
                - class: str
        """
        total_start = time.perf_counter()

        # Parse GeoJSON
        parse_start = time.perf_counter()
        df, has_tile_id = self._parse_geojson()
        parse_time = time.perf_counter() - parse_start

        # Spatial matching if no tile_id
        spatial_time = 0.0
        if not has_tile_id:
            spatial_start = time.perf_counter()
            df = self._spatial_match(df)
            spatial_time = time.perf_counter() - spatial_start

        # Fetch embeddings
        fetch_start = time.perf_counter()
        tile_ids = df["tile_id"].tolist()
        embeddings_dict = self._fetch_embeddings(tile_ids)
        fetch_time = time.perf_counter() - fetch_start

        # Add embeddings to dataframe
        df["embedding"] = df["tile_id"].map(embeddings_dict)

        # Remove any rows where embedding fetch failed
        missing_embeddings = df["embedding"].isna()
        if missing_embeddings.any():
            missing_count = missing_embeddings.sum()
            missing_ids = df.loc[missing_embeddings, "tile_id"].tolist()
            raise ValueError(
                f"Missing embeddings for {missing_count} tiles: {missing_ids[:10]}..."
            )

        # Stratified split
        split_start = time.perf_counter()
        train_df, test_df = self._stratified_split(df, test_fraction, random_state)
        split_time = time.perf_counter() - split_start

        total_time = time.perf_counter() - total_start

        timing = LoaderTiming(
            parse_geojson_sec=parse_time,
            spatial_match_sec=spatial_time,
            fetch_embeddings_sec=fetch_time,
            stratified_split_sec=split_time,
            total_sec=total_time,
        )

        return train_df, test_df, timing

    def _parse_geojson(self) -> Tuple[pd.DataFrame, bool]:
        """
        Parse GeoJSON file and extract properties and geometry.

        Returns:
            Tuple of (DataFrame, has_tile_id) where DataFrame has columns:
                - tile_id (if present in properties, or looked up from id)
                - label
                - class
                - lon, lat (if geometry present and no tile_id/id)
        """
        with open(self.geojson_path, "r") as f:
            geojson_data = json.load(f)

        records = []
        has_tile_id = False
        db_ids_to_lookup = []

        for i, feature in enumerate(geojson_data["features"]):
            props = feature["properties"]
            record = {
                "label": props["label"],
                "class": props["class"],
            }

            # Check for tile_id first
            if "tile_id" in props:
                record["tile_id"] = props["tile_id"]
                has_tile_id = True
            # Check for numeric id (DuckDB row id)
            elif "id" in props:
                record["db_id"] = int(props["id"])
                db_ids_to_lookup.append(int(props["id"]))
                has_tile_id = True  # Will be resolved via lookup
            else:
                # Extract coordinates from geometry for spatial matching
                geom = feature["geometry"]
                if geom["type"] != "Point":
                    raise ValueError(
                        f"Feature {i}: Expected Point geometry for spatial matching, "
                        f"got {geom['type']}"
                    )
                record["lon"] = geom["coordinates"][0]
                record["lat"] = geom["coordinates"][1]

            records.append(record)

        df = pd.DataFrame(records)

        # If we have db_ids to lookup, resolve them to tile_ids
        if db_ids_to_lookup:
            print(f"Looking up tile_ids for {len(db_ids_to_lookup)} database IDs...")
            id_to_tile = self._lookup_tile_ids_from_db_ids(db_ids_to_lookup)
            # Only map for rows that have db_id (not for rows that already have tile_id)
            if "db_id" in df.columns:
                needs_lookup = df["db_id"].notna()
                df.loc[needs_lookup, "tile_id"] = df.loc[needs_lookup, "db_id"].map(
                    id_to_tile
                )
                # Check for any missing lookups
                missing = df["tile_id"].isna() & df["db_id"].notna()
                if missing.any():
                    missing_ids = df.loc[missing, "db_id"].tolist()[:10]
                    raise ValueError(
                        f"Could not find tile_ids for db_ids: {missing_ids}..."
                    )
                df = df.drop(columns=["db_id"])

        # Validate data types
        df["label"] = df["label"].astype(int)
        df["class"] = df["class"].astype(str)

        # Validate label values
        valid_labels = df["label"].isin([0, 1])
        if not valid_labels.all():
            invalid_labels = df.loc[~valid_labels, "label"].unique()
            raise ValueError(
                f"Invalid label values found: {invalid_labels}. Must be 0 or 1."
            )

        return df, has_tile_id

    def _lookup_tile_ids_from_db_ids(self, db_ids: List[int]) -> Dict[int, str]:
        """
        Look up tile_ids from DuckDB row ids.

        Args:
            db_ids: List of DuckDB row IDs

        Returns:
            Dictionary mapping db_id -> tile_id
        """
        ids_str = ",".join(str(i) for i in db_ids)
        query = f"""
            SELECT id, tile_id
            FROM geo_embeddings
            WHERE id IN ({ids_str})
        """
        result = self.conn.execute(query).fetchall()
        return {row[0]: row[1] for row in result}

    def _spatial_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find nearest tile_id for each point using spatial matching.

        Uses UTM projection for accurate distance calculation in meters.

        Args:
            df: DataFrame with lon, lat columns

        Returns:
            DataFrame with tile_id column added
        """
        # Determine UTM zone from centroid of points
        center_lon = df["lon"].mean()
        center_lat = df["lat"].mean()
        utm_zone = int(((center_lon + 180) / 6) + 1)
        hemisphere = "N" if center_lat >= 0 else "S"
        utm_epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

        print(
            f"Spatial matching {len(df)} points using UTM zone {utm_zone}{hemisphere}..."
        )

        # Build query to find nearest point for each input point
        # Using DuckDB spatial functions with UTM projection for accurate distances
        results = []

        for idx, row in df.iterrows():
            lon, lat = row["lon"], row["lat"]

            # Query finds nearest tile by distance in meters (UTM projection)
            # Use ST_Distance for ordering since <-> operator doesn't work with GEOMETRY
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

            result = self.conn.execute(query).fetchone()
            if result is None:
                raise ValueError(f"No nearby tile found for point ({lon}, {lat})")

            tile_id, distance_m = result
            results.append({"idx": idx, "tile_id": tile_id, "distance_m": distance_m})

        # Add tile_id to dataframe
        match_df = pd.DataFrame(results).set_index("idx")
        df["tile_id"] = match_df["tile_id"]
        df["match_distance_m"] = match_df["distance_m"]

        # Report statistics
        print(f"  Mean match distance: {df['match_distance_m'].mean():.1f}m")
        print(f"  Max match distance: {df['match_distance_m'].max():.1f}m")

        # Warn if any matches are far
        far_matches = df[df["match_distance_m"] > 500]
        if len(far_matches) > 0:
            print(f"  WARNING: {len(far_matches)} points matched >500m away")

        return df

    def _fetch_embeddings(self, tile_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetch embeddings from DuckDB for given tile_ids.

        Casts UTINYINT[384] to FLOAT and returns as numpy arrays.

        Args:
            tile_ids: List of tile IDs to fetch

        Returns:
            Dictionary mapping tile_id -> embedding (np.ndarray of float32)
        """
        # Build query with string-escaped tile_ids (single quotes for SQL strings)
        escaped_ids = ",".join(f"'{tid}'" for tid in tile_ids)
        query = f"""
            SELECT tile_id, embedding
            FROM geo_embeddings
            WHERE tile_id IN ({escaped_ids})
        """

        result = self.conn.execute(query).fetchall()

        embeddings_dict = {}
        for tile_id, embedding in result:
            # Convert UTINYINT array to float32 numpy array
            # DuckDB returns the embedding as a list of integers
            embedding_array = np.array(embedding, dtype=np.float32)
            embeddings_dict[tile_id] = embedding_array

        return embeddings_dict

    def _stratified_split(
        self, df: pd.DataFrame, test_fraction: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split with equal negatives per class.

        Strategy:
        1. Split positives (label=1) using standard train/test split
        2. Split negatives (label=0) ensuring equal representation from each class
        3. For negative test set, sample floor(test_fraction * min_class_count) from each class

        Args:
            df: DataFrame with tile_id, embedding, label, class columns
            test_fraction: Fraction of data for test set
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        # Separate positives and negatives
        positives = df[df["label"] == 1].copy()
        negatives = df[df["label"] == 0].copy()

        # Split positives using standard stratified split by class
        if len(positives) > 0:
            pos_classes = positives["class"].value_counts()
            pos_test_dfs = []
            pos_train_dfs = []

            for class_name in pos_classes.index:
                class_data = positives[positives["class"] == class_name]
                n_test = max(1, int(len(class_data) * test_fraction))

                # Shuffle and split
                shuffled = class_data.sample(frac=1, random_state=random_state)
                pos_test_dfs.append(shuffled.iloc[:n_test])
                pos_train_dfs.append(shuffled.iloc[n_test:])

            pos_test = pd.concat(pos_test_dfs, ignore_index=True)
            pos_train = pd.concat(pos_train_dfs, ignore_index=True)
        else:
            pos_test = pd.DataFrame(columns=df.columns)
            pos_train = pd.DataFrame(columns=df.columns)

        # Split negatives with equal samples per class
        if len(negatives) > 0:
            neg_classes = negatives.groupby("class")

            # Find minimum class count
            class_counts = negatives["class"].value_counts()
            min_class_count = class_counts.min()

            # Calculate samples per class for test set
            samples_per_class_test = int(test_fraction * min_class_count)

            if samples_per_class_test == 0:
                # If test fraction is too small, take at least 1 per class if possible
                samples_per_class_test = 1

            neg_test_dfs = []
            neg_train_dfs = []

            for class_name, class_data in neg_classes:
                # Shuffle class data
                shuffled = class_data.sample(frac=1, random_state=random_state)

                # Take equal samples for test
                n_test = min(samples_per_class_test, len(shuffled))
                neg_test_dfs.append(shuffled.iloc[:n_test])
                neg_train_dfs.append(shuffled.iloc[n_test:])

            neg_test = pd.concat(neg_test_dfs, ignore_index=True)
            neg_train = pd.concat(neg_train_dfs, ignore_index=True)
        else:
            neg_test = pd.DataFrame(columns=df.columns)
            neg_train = pd.DataFrame(columns=df.columns)

        # Combine positives and negatives
        train_df = pd.concat([pos_train, neg_train], ignore_index=True)
        test_df = pd.concat([pos_test, neg_test], ignore_index=True)

        # Shuffle final datasets
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

        return train_df, test_df
