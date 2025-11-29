"""
Data loader for classification training.

Loads GeoJSON training data and fetches corresponding embeddings from DuckDB.
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
    fetch_embeddings_sec: float
    stratified_split_sec: float
    total_sec: float


class ClassificationDataLoader:
    """
    Loads training data from GeoJSON and fetches embeddings from DuckDB.

    Supports stratified train/test split with equal negatives per class.
    """

    def __init__(self, duckdb_connection: duckdb.DuckDBPyConnection, geojson_path: str):
        """
        Initialize the data loader.

        Args:
            duckdb_connection: Active DuckDB connection to database with geo_embeddings table
            geojson_path: Path to GeoJSON file with tile_id, label, class properties
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
        df = self._parse_geojson()
        parse_time = time.perf_counter() - parse_start

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
            fetch_embeddings_sec=fetch_time,
            stratified_split_sec=split_time,
            total_sec=total_time,
        )

        return train_df, test_df, timing

    def _parse_geojson(self) -> pd.DataFrame:
        """
        Parse GeoJSON file and extract tile_id, label, class from properties.

        Returns:
            DataFrame with columns: tile_id, label, class
        """
        with open(self.geojson_path, "r") as f:
            geojson_data = json.load(f)

        records = []
        for feature in geojson_data["features"]:
            props = feature["properties"]
            records.append(
                {
                    "tile_id": props["tile_id"],
                    "label": props["label"],
                    "class": props["class"],
                }
            )

        df = pd.DataFrame(records)

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
        # Build parameterized query
        placeholders = ",".join(["?" for _ in tile_ids])
        query = f"""
            SELECT tile_id, embedding
            FROM geo_embeddings
            WHERE tile_id IN ({placeholders})
        """

        result = self.conn.execute(query, tile_ids).fetchall()

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
