"""Data loader for classification training."""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import pandas as pd
import geopandas as gpd
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
    """

    def __init__(self, duckdb_connection: duckdb.DuckDBPyConnection, geojson_path: str):
        """
        Initialize the data loader.

        Parameters
        ----------
        duckdb_connection : duckdb.DuckDBPyConnection
            Active DuckDB connection with geo_embeddings table
        geojson_path : str
            Path to GeoJSON file with label, class properties
        """
        self.conn = duckdb_connection
        self.geojson_path = geojson_path
        self._id_column: str | None = None

    def _get_id_column(self) -> str:
        """Detect which ID column exists in the geo_embeddings table."""
        if self._id_column is not None:
            return self._id_column

        result = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'geo_embeddings' AND column_name IN ('tile_id', 'id')"
        ).fetchall()
        columns = {row[0] for row in result}
        self._id_column = "tile_id" if "tile_id" in columns else "id"
        return self._id_column

    def load(
        self, test_fraction: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, LoaderTiming]:
        """
        Load and split training data.

        Parameters
        ----------
        test_fraction : float
            Fraction of data to use for test set (0.0 to 1.0)
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, LoaderTiming]
            train_df, test_df with columns: tile_id, embedding, label, class
        """
        total_start = time.perf_counter()

        parse_start = time.perf_counter()
        df, has_tile_id = self._parse_geojson()
        parse_time = time.perf_counter() - parse_start

        spatial_time = 0.0
        if not has_tile_id:
            spatial_start = time.perf_counter()
            df = self._spatial_match(df)
            spatial_time = time.perf_counter() - spatial_start

        fetch_start = time.perf_counter()
        tile_ids = df["tile_id"].tolist()
        embeddings_dict = self._fetch_embeddings(tile_ids)
        fetch_time = time.perf_counter() - fetch_start

        # Normalize tile_ids to strings for consistent mapping
        id_col = self._get_id_column()
        if id_col == "id":
            df["tile_id"] = df["tile_id"].apply(lambda x: str(int(float(x))))
        else:
            df["tile_id"] = df["tile_id"].astype(str)

        df["embedding"] = df["tile_id"].map(embeddings_dict)

        missing_embeddings = df["embedding"].isna()
        if missing_embeddings.any():
            missing_count = missing_embeddings.sum()
            missing_ids = df.loc[missing_embeddings, "tile_id"].tolist()
            raise ValueError(
                f"Missing embeddings for {missing_count} tiles: {missing_ids[:10]}..."
            )

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
        """Parse GeoJSON file and extract properties and geometry."""
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

            if "tile_id" in props:
                record["tile_id"] = props["tile_id"]
                has_tile_id = True
            elif "id" in props:
                record["db_id"] = int(props["id"])
                db_ids_to_lookup.append(int(props["id"]))
                has_tile_id = True
            else:
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

        if db_ids_to_lookup:
            print(f"Looking up tile_ids for {len(db_ids_to_lookup)} database IDs...")
            id_to_tile = self._lookup_tile_ids_from_db_ids(db_ids_to_lookup)
            if "db_id" in df.columns:
                needs_lookup = df["db_id"].notna()
                df.loc[needs_lookup, "tile_id"] = df.loc[needs_lookup, "db_id"].map(
                    id_to_tile
                )
                missing = df["tile_id"].isna() & df["db_id"].notna()
                if missing.any():
                    missing_ids = df.loc[missing, "db_id"].tolist()[:10]
                    raise ValueError(
                        f"Could not find tile_ids for db_ids: {missing_ids}..."
                    )
                df = df.drop(columns=["db_id"])

        df["label"] = df["label"].astype(int)
        df["class"] = df["class"].astype(str)

        valid_labels = df["label"].isin([0, 1])
        if not valid_labels.all():
            invalid_labels = df.loc[~valid_labels, "label"].unique()
            raise ValueError(
                f"Invalid label values found: {invalid_labels}. Must be 0 or 1."
            )

        return df, has_tile_id

    def _lookup_tile_ids_from_db_ids(self, db_ids: List[int]) -> Dict[int, str]:
        """Look up tile_ids from DuckDB row ids."""
        id_col = self._get_id_column()
        ids_str = ",".join(str(i) for i in db_ids)
        query = f"""
            SELECT id, {id_col} as tile_id
            FROM geo_embeddings
            WHERE id IN ({ids_str})
        """
        result = self.conn.execute(query).fetchall()
        return {row[0]: row[1] for row in result}

    def _spatial_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find nearest tile_id for each point using spatial matching."""
        id_col = self._get_id_column()
        center_lon = df["lon"].mean()
        center_lat = df["lat"].mean()
        utm_zone = int(((center_lon + 180) / 6) + 1)
        hemisphere = "N" if center_lat >= 0 else "S"
        utm_epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

        print(
            f"Spatial matching {len(df)} points using UTM zone {utm_zone}{hemisphere}..."
        )

        results = []

        for idx, row in df.iterrows():
            lon, lat = row["lon"], row["lat"]

            query = f"""
                SELECT
                    {id_col} as tile_id,
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

        match_df = pd.DataFrame(results).set_index("idx")
        df["tile_id"] = match_df["tile_id"]
        df["match_distance_m"] = match_df["distance_m"]

        print(f"  Mean match distance: {df['match_distance_m'].mean():.1f}m")
        print(f"  Max match distance: {df['match_distance_m'].max():.1f}m")

        far_matches = df[df["match_distance_m"] > 500]
        if len(far_matches) > 0:
            print(f"  WARNING: {len(far_matches)} points matched >500m away")

        return df

    def _format_ids_for_query(self, tile_ids: List[str]) -> str:
        """Format tile IDs for SQL IN clause, handling both string and int columns."""
        id_col = self._get_id_column()
        if id_col == "id":
            # Integer column - convert to int, no quotes
            return ",".join(str(int(float(tid))) for tid in tile_ids)
        else:
            # String column - use quotes
            return ",".join(f"'{tid}'" for tid in tile_ids)

    def _fetch_embeddings(self, tile_ids: List[str]) -> Dict[str, np.ndarray]:
        """Fetch embeddings from DuckDB for given tile_ids."""
        id_col = self._get_id_column()
        formatted_ids = self._format_ids_for_query(tile_ids)
        query = f"""
            SELECT {id_col} as tile_id, embedding
            FROM geo_embeddings
            WHERE {id_col} IN ({formatted_ids})
        """

        result = self.conn.execute(query).fetchall()

        embeddings_dict = {}
        for tile_id, embedding in result:
            embeddings_dict[str(tile_id)] = np.array(embedding, dtype=np.float32)

        return embeddings_dict

    def _fetch_geometries(self, tile_ids: List[str]) -> gpd.GeoSeries:
        """Fetch point geometries from DuckDB for given tile_ids."""
        id_col = self._get_id_column()
        formatted_ids = self._format_ids_for_query(tile_ids)
        query = f"""
            SELECT {id_col} as tile_id, ST_AsText(geometry) as geometry_wkt
            FROM geo_embeddings
            WHERE {id_col} IN ({formatted_ids})
        """
        result = self.conn.execute(query).fetchdf()
        result["geometry"] = gpd.GeoSeries.from_wkt(result["geometry_wkt"])
        tile_to_geom = dict(zip(result["tile_id"].astype(str), result["geometry"]))
        return gpd.GeoSeries(
            [tile_to_geom.get(str(tid)) for tid in tile_ids], crs="EPSG:4326"
        )

    def load_for_cv(
        self, random_state: int = 42
    ) -> Tuple[pd.DataFrame, gpd.GeoSeries, LoaderTiming]:
        """
        Load all training data without split, with geometries for spatial CV.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        Tuple[pd.DataFrame, gpd.GeoSeries, LoaderTiming]
            df with columns: tile_id, embedding, label, class
            geometries: Point geometries aligned with df rows
            timing: Loading timing info
        """
        total_start = time.perf_counter()

        parse_start = time.perf_counter()
        df, has_tile_id = self._parse_geojson()
        parse_time = time.perf_counter() - parse_start

        spatial_time = 0.0
        if not has_tile_id:
            spatial_start = time.perf_counter()
            df = self._spatial_match(df)
            spatial_time = time.perf_counter() - spatial_start

        fetch_start = time.perf_counter()
        tile_ids = df["tile_id"].tolist()
        embeddings_dict = self._fetch_embeddings(tile_ids)
        geometries = self._fetch_geometries(tile_ids)
        fetch_time = time.perf_counter() - fetch_start

        # Normalize tile_ids to strings for consistent mapping
        id_col = self._get_id_column()
        if id_col == "id":
            df["tile_id"] = df["tile_id"].apply(lambda x: str(int(float(x))))
        else:
            df["tile_id"] = df["tile_id"].astype(str)

        df["embedding"] = df["tile_id"].map(embeddings_dict)

        missing_embeddings = df["embedding"].isna()
        if missing_embeddings.any():
            missing_count = missing_embeddings.sum()
            missing_ids = df.loc[missing_embeddings, "tile_id"].tolist()
            raise ValueError(
                f"Missing embeddings for {missing_count} tiles: {missing_ids[:10]}..."
            )

        shuffle_idx = df.sample(frac=1, random_state=random_state).index
        df = df.loc[shuffle_idx].reset_index(drop=True)
        geometries = geometries.iloc[shuffle_idx].reset_index(drop=True)

        total_time = time.perf_counter() - total_start

        timing = LoaderTiming(
            parse_geojson_sec=parse_time,
            spatial_match_sec=spatial_time,
            fetch_embeddings_sec=fetch_time,
            stratified_split_sec=0.0,
            total_sec=total_time,
        )

        return df, geometries, timing

    def _stratified_split(
        self, df: pd.DataFrame, test_fraction: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform stratified train/test split with equal negatives per class."""
        positives = df[df["label"] == 1].copy()
        negatives = df[df["label"] == 0].copy()

        if len(positives) > 0:
            pos_classes = positives["class"].value_counts()
            pos_test_dfs = []
            pos_train_dfs = []

            for class_name in pos_classes.index:
                class_data = positives[positives["class"] == class_name]
                n_test = max(1, int(len(class_data) * test_fraction))

                shuffled = class_data.sample(frac=1, random_state=random_state)
                pos_test_dfs.append(shuffled.iloc[:n_test])
                pos_train_dfs.append(shuffled.iloc[n_test:])

            pos_test = pd.concat(pos_test_dfs, ignore_index=True)
            pos_train = pd.concat(pos_train_dfs, ignore_index=True)
        else:
            pos_test = pd.DataFrame(columns=df.columns)
            pos_train = pd.DataFrame(columns=df.columns)

        if len(negatives) > 0:
            neg_classes = negatives.groupby("class")

            class_counts = negatives["class"].value_counts()
            min_class_count = class_counts.min()

            samples_per_class_test = int(test_fraction * min_class_count)

            if samples_per_class_test == 0:
                samples_per_class_test = 1

            neg_test_dfs = []
            neg_train_dfs = []

            for class_name, class_data in neg_classes:
                shuffled = class_data.sample(frac=1, random_state=random_state)

                n_test = min(samples_per_class_test, len(shuffled))
                neg_test_dfs.append(shuffled.iloc[:n_test])
                neg_train_dfs.append(shuffled.iloc[n_test:])

            neg_test = pd.concat(neg_test_dfs, ignore_index=True)
            neg_train = pd.concat(neg_train_dfs, ignore_index=True)
        else:
            neg_test = pd.DataFrame(columns=df.columns)
            neg_train = pd.DataFrame(columns=df.columns)

        train_df = pd.concat([pos_train, neg_train], ignore_index=True)
        test_df = pd.concat([pos_test, neg_test], ignore_index=True)

        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

        return train_df, test_df
