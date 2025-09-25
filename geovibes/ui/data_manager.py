"""Data access and configuration helpers for the GeoVibes UI."""

from __future__ import annotations

import csv
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import duckdb
import ee
import faiss
import geopandas as gpd
import shapely.geometry

from geovibes.ee_tools import initialize_ee_with_credentials
from geovibes.ui_config import BasemapConfig, DatabaseConstants, GeoVibesConfig
from geovibes.utils import get_database_centroid, list_databases_in_directory

from .utils import log_to_file, parse_env_flag, prepare_ids_for_query


class DataManager:
    """Encapsulates configuration, database, and FAISS operations."""

    def __init__(
        self,
        *,
        duckdb_path: Optional[str] = None,
        duckdb_directory: Optional[str] = None,
        boundary_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        gcp_project: Optional[str] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None,
        baselayer_url: Optional[str] = None,
        enable_ee: Optional[bool] = None,
        disable_ee: bool = False,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.baselayer_url = baselayer_url or BasemapConfig.BASEMAP_TILES["MAPTILER"]

        # Configuration and Earth Engine toggles
        self.config = self._load_config(
            duckdb_path=duckdb_path,
            duckdb_directory=duckdb_directory,
            boundary_path=boundary_path,
            start_date=start_date,
            end_date=end_date,
            gcp_project=gcp_project,
            config=config,
            config_path=config_path,
        )

        self.enable_ee = self._resolve_enable_ee(enable_ee, disable_ee)
        self.ee_available = False
        if self.enable_ee:
            self.ee_available = initialize_ee_with_credentials(
                self.config.gcp_project, verbose=self.verbose
            )
        elif self.verbose:
            print("ℹ️ Earth Engine disabled - running with basic basemaps only")

        self.geometries_dir = self._resolve_geometries_directory()
        self.local_database_directory = self._resolve_local_database_directory()
        self.manifest_entries: List[Dict[str, str]] = []

        self.available_databases = self._discover_databases()
        if not self.available_databases:
            raise FileNotFoundError(
                "No downloaded models found. Provide duckdb_path/duckdb_directory or run prep_data.py."
            )

        self.available_databases = sorted(
            self.available_databases,
            key=lambda entry: entry.get(
                "display_name", os.path.basename(entry["db_path"])
            ),
        )
        self.database_info_by_path = {
            entry["db_path"]: entry for entry in self.available_databases
        }

        self.current_database_info = self.available_databases[0]
        self.current_database_path = self.current_database_info["db_path"]
        self.current_faiss_path = self.current_database_info.get("faiss_path")
        self.current_geometry_path = self.current_database_info.get("geometry_path")
        self.effective_boundary_path = None
        self.ee_boundary = None

        # Manage DuckDB connection
        if duckdb_connection is None:
            self.duckdb_connection = self._connect_duckdb(self.current_database_path)
            self._owns_connection = True
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False

        self._apply_duckdb_settings(self.current_database_path)

        # Load FAISS index
        if not self.current_faiss_path:
            raise ValueError("Could not find a FAISS index for the selected database.")
        if self.verbose:
            print(f"🧠 Loading FAISS index from: {self.current_faiss_path}")
        self.faiss_index = faiss.read_index(self.current_faiss_path)
        if self.verbose:
            print(f"✅ FAISS index loaded. Contains {self.faiss_index.ntotal} vectors.")

        # Detect embedding dimension
        self.embedding_dim = self._detect_embedding_dim()

        # Warm up if needed
        if DatabaseConstants.is_gcs_path(self.current_database_path):
            self._warm_up_gcs_database()

        # Derive map centering data
        self.effective_boundary_path, (self.center_y, self.center_x) = (
            self._setup_boundary_and_center()
        )
        self._update_ee_boundary()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_config(
        self,
        *,
        duckdb_path: Optional[str],
        duckdb_directory: Optional[str],
        boundary_path: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        gcp_project: Optional[str],
        config: Optional[Dict],
        config_path: Optional[str],
    ) -> GeoVibesConfig:
        if config_path is not None:
            cfg = GeoVibesConfig.from_file(config_path)
        elif config is not None:
            cfg = GeoVibesConfig.from_dict(config)
        else:
            cfg = GeoVibesConfig(
                duckdb_path=duckdb_path,
                duckdb_directory=duckdb_directory,
                boundary_path=boundary_path,
                start_date=start_date or "2024-01-01",
                end_date=end_date or "2025-01-01",
                gcp_project=gcp_project,
            )

        if hasattr(cfg, "validate"):
            try:
                cfg.validate()
            except Exception as exc:  # pragma: no cover - defensive logging
                if self.verbose:
                    print(f"⚠️ Config validation skipped: {exc}")
        return cfg

    def _resolve_enable_ee(
        self, enable_ee: Optional[bool], disable_ee: bool
    ) -> bool:
        env_enable = parse_env_flag(os.getenv("GEOVIBES_ENABLE_EE"))
        env_disable = parse_env_flag(os.getenv("GEOVIBES_DISABLE_EE"))

        ee_opt_in = enable_ee
        if disable_ee:
            ee_opt_in = False
        elif ee_opt_in is None and env_disable is True:
            ee_opt_in = False
        elif ee_opt_in is None and env_enable is True:
            ee_opt_in = True
        elif ee_opt_in is None:
            ee_opt_in = getattr(self.config, "enable_ee", False)
        return bool(ee_opt_in)

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_root() -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parent.parent

    def _resolve_manifest_path(self) -> Optional[str]:
        override = os.getenv("GEOVIBES_MANIFEST_PATH")
        if override:
            candidate = pathlib.Path(override).expanduser()
            if candidate.exists():
                return str(candidate)

        manifest_path = self._project_root() / "manifest.csv"
        return str(manifest_path) if manifest_path.exists() else None

    def _resolve_geometries_directory(self) -> Optional[str]:
        override = os.getenv("GEOVIBES_GEOMETRIES_DIR")
        if override:
            return str(pathlib.Path(override).expanduser())
        return str(self._project_root() / "geometries")

    def _resolve_local_database_directory(self) -> str:
        override = os.getenv("GEOVIBES_LOCAL_DB_DIR")
        if override:
            return str(pathlib.Path(override).expanduser())
        return str(self._project_root() / "local_databases")

    def _discover_databases(self) -> List[Dict[str, str]]:
        discovered: List[Dict[str, str]] = []

        if getattr(self.config, "duckdb_path", None):
            db_path = self.config.duckdb_path
            faiss_path = self._infer_faiss_from_db(db_path)
            if not faiss_path:
                if self.verbose:
                    print(
                        f"⚠️  Could not locate FAISS index for {db_path}. Skipping."
                    )
            else:
                discovered.append(
                    {
                        "db_path": db_path,
                        "faiss_path": faiss_path,
                        "display_name": os.path.basename(db_path),
                        "geometry_path": getattr(self.config, "boundary_path", None),
                    }
                )
                return discovered

        if getattr(self.config, "duckdb_directory", None):
            directory_entries = list_databases_in_directory(
                self.config.duckdb_directory, verbose=self.verbose
            )
            for entry in directory_entries:
                if not entry.get("faiss_path"):
                    if self.verbose:
                        print(
                            f"⚠️  Missing FAISS index for {entry['db_path']}. Skipping."
                        )
                    continue
                entry.setdefault("display_name", os.path.basename(entry["db_path"]))
                entry.setdefault("geometry_path", getattr(self.config, "boundary_path", None))
                discovered.append(entry)
            for entry in discovered:
                entry.setdefault("geometry_path", getattr(self.config, "boundary_path", None))
            if discovered:
                return discovered

        # Fallback to manifest-driven discovery
        manifest_path = self._resolve_manifest_path()
        if manifest_path is None:
            return discovered

        self.manifest_entries = self._load_manifest_entries(manifest_path)
        if not self.manifest_entries:
            return discovered

        if self.verbose:
            print(
                f"📄 Loaded {len(self.manifest_entries)} manifest entries from {manifest_path}"
            )

        discovered.extend(
            self._discover_available_models(
                self.local_database_directory, self.manifest_entries
            )
        )
        return discovered

    def _infer_faiss_from_db(self, db_path: str) -> Optional[str]:
        candidate = pathlib.Path(db_path)
        base = candidate.stem
        name_candidates = {base}
        if base.endswith("_metadata"):
            name_candidates.add(base[: -len("_metadata")])

        patterns = []
        for name in name_candidates:
            patterns.extend(
                [
                    f"{name}.index",
                    f"{name}_faiss.index",
                    f"{name}_faiss*.index",
                    f"{name}*.index",
                ]
            )

        for pattern in patterns:
            matches = sorted(candidate.parent.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest_entries(self, manifest_path: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        try:
            with open(manifest_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    entries.append(
                        {
                            k: (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items()
                        }
                    )
        except Exception as exc:  # pragma: no cover
            if self.verbose:
                print(f"⚠️  Failed to read manifest at {manifest_path}: {exc}")
        return entries

    def _discover_available_models(
        self, directory_path: str, manifest_rows: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if not directory_path:
            return []

        dir_path = pathlib.Path(directory_path).expanduser()
        if not dir_path.exists():
            if self.verbose:
                print(f"⚠️  Database directory not found: {directory_path}")
            return []

        manifest_lookup = {
            (row.get("model_name") or "").strip(): row
            for row in manifest_rows
            if row.get("model_name")
        }

        discovered: List[Dict[str, str]] = []
        seen_db_paths = set()

        for model_name, row in manifest_lookup.items():
            artifacts = self._locate_model_artifacts(dir_path, model_name)
            if not artifacts:
                if self.verbose:
                    print(f"⚠️  Model files not found locally for {model_name}")
                continue

            if artifacts["db_path"] in seen_db_paths:
                continue

            region = (row.get("region") or "").strip() or None
            geometry_path = self._resolve_geometry_path(region)
            entry = {
                "model_name": model_name,
                "region": region,
                "db_path": artifacts["db_path"],
                "faiss_path": artifacts["faiss_path"],
                "display_name": self._format_model_display_name(region, model_name),
                "geometry_path": geometry_path,
            }
            discovered.append(entry)
            seen_db_paths.add(entry["db_path"])

        return discovered

    def _locate_model_artifacts(
        self, root_directory: pathlib.Path, model_name: str
    ) -> Optional[Dict[str, str]]:
        candidate_dirs = [root_directory / model_name, root_directory]
        for candidate in candidate_dirs:
            if not candidate.exists():
                continue

            db_path = self._match_single_file(
                candidate,
                [
                    (f"{model_name}_metadata.db", f"{model_name}_metadata.db"),
                    (f"{model_name}_metadata" + "*.db", f"{model_name}_metadata.db"),
                    (f"{model_name}.db", f"{model_name}.db"),
                ],
            )

            if not db_path:
                continue

            faiss_path = self._match_single_file(
                candidate,
                [
                    (f"{model_name}_faiss" + "*.index", None),
                    (f"{model_name}.index", f"{model_name}.index"),
                ],
            )

            if db_path and faiss_path:
                return {"db_path": db_path, "faiss_path": faiss_path}

        return None

    def _match_single_file(
        self, directory: pathlib.Path, pattern_specs: List[Tuple[str, Optional[str]]]
    ) -> Optional[str]:
        for pattern, preferred in pattern_specs:
            matches = sorted(directory.glob(pattern))
            selected = self._select_preferred_path(matches, preferred)
            if selected:
                return str(selected)
        return None

    @staticmethod
    def _select_preferred_path(
        matches: List[pathlib.Path], preferred_name: Optional[str] = None
    ) -> Optional[pathlib.Path]:
        if not matches:
            return None

        if preferred_name:
            for match in matches:
                if match.name == preferred_name:
                    return match

        matches = sorted(
            matches,
            key=lambda path: (
                DataManager._has_numeric_suffix(path.stem),
                len(path.name),
                path.name,
            ),
        )
        return matches[0]

    @staticmethod
    def _has_numeric_suffix(stem: str) -> int:
        return 1 if re.search(r"_\d+$", stem) else 0

    def _resolve_geometry_path(self, region: Optional[str]) -> Optional[str]:
        if not region or not self.geometries_dir:
            return None

        geom_dir = pathlib.Path(self.geometries_dir)
        candidates = [
            geom_dir / f"{region}.geojson",
            geom_dir / f"{region}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _format_model_display_name(region: Optional[str], model_name: str) -> str:
        if region:
            return f"{region} / {model_name}"
        return model_name

    # ------------------------------------------------------------------
    # DuckDB helpers
    # ------------------------------------------------------------------

    def _connect_duckdb(self, database_path: str) -> duckdb.DuckDBPyConnection:
        if DatabaseConstants.is_gcs_path(database_path) and self.verbose:
            print(f"🌐 Connecting to GCS database: {database_path}")
            if os.getenv("GCS_ACCESS_KEY_ID"):
                print("🔑 Using HMAC key authentication")
            else:
                print("🔑 Using default Google Cloud authentication")
        elif self.verbose:
            print(f"💾 Connecting to local database: {database_path}")

        try:
            connection = DatabaseConstants.setup_duckdb_connection(
                database_path, read_only=True
            )
            if self.verbose:
                print("✅ Database connection established successfully")
            return connection
        except Exception as exc:
            if DatabaseConstants.is_gcs_path(database_path):
                error_msg = f"Failed to connect to GCS database: {exc}"
                if (
                    "authentication" in str(exc).lower()
                    or "forbidden" in str(exc).lower()
                ):
                    error_msg += "\n💡 Check your GCS authentication setup (see GCS_SETUP.md)"
                raise RuntimeError(error_msg)
            raise RuntimeError(f"Failed to connect to local database: {exc}")

    def _apply_duckdb_settings(self, database_path: Optional[str]) -> None:
        for query in DatabaseConstants.get_memory_setup_queries():
            self.duckdb_connection.execute(query)
        try:
            self.duckdb_connection.execute("SET enable_progress_bar=false")
            self.duckdb_connection.execute("SET enable_profiling=false")
            self.duckdb_connection.execute("SET enable_object_cache=false")
            if self.verbose:
                print("✅ Progress bar and profiling disabled")
        except Exception:  # pragma: no cover - optional settings
            pass

        if database_path:
            extension_queries = DatabaseConstants.get_extension_setup_queries(
                database_path
            )
            for query in extension_queries:
                try:
                    self.duckdb_connection.execute(query)
                    if self.verbose:
                        if "httpfs" in query:
                            print("📦 httpfs extension loaded for GCS support")
                        elif "spatial" in query:
                            print("🗺️  spatial extension loaded for geometry support")
                except Exception as exc:
                    raise RuntimeError(f"Failed to load required extension: {exc}")

    def _detect_embedding_dim(self) -> int:
        try:
            embedding_dim = DatabaseConstants.detect_embedding_dimension(
                self.duckdb_connection
            )
            if self.verbose:
                print(f"🔍 Detected embedding dimension: {embedding_dim}")
            return embedding_dim
        except ValueError as exc:
            if self.verbose:
                print(f"⚠️ Could not detect embedding dimension: {exc}")
                print("⚠️ Using default dimension of 384")
            return 384

    def _warm_up_gcs_database(self) -> None:
        try:
            if self.verbose:
                print("🔧 Optimizing database connection...")

            first_point_query = """
            SELECT CAST(embedding AS FLOAT[]) as embedding 
            FROM geo_embeddings 
            WHERE embedding IS NOT NULL 
            LIMIT 1
            """
            result = self.duckdb_connection.execute(first_point_query).fetchone()
            if not result or not result[0]:
                if self.verbose:
                    print("⚠️  No embeddings found for warm-up")
                return

            first_embedding = result[0]
            sql = DatabaseConstants.get_similarity_search_light_query(
                self.embedding_dim
            )
            query_params = [first_embedding, 100]
            self.duckdb_connection.execute(sql, query_params).fetchall()
            if self.verbose:
                print("✅ Database optimization completed")
        except Exception as exc:
            if self.verbose:
                print(f"⚠️  Database warm-up failed: {exc}")

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def _setup_boundary_and_center(self):
        boundary_path = self.current_geometry_path
        if (
            not boundary_path
            and self.current_database_info
            and self.current_database_info.get("geometry_path")
        ):
            boundary_path = self.current_database_info.get("geometry_path")

        if boundary_path:
            try:
                boundary_gdf = gpd.read_file(boundary_path)
                center_y, center_x = (
                    boundary_gdf.geometry.iloc[0].centroid.y,
                    boundary_gdf.geometry.iloc[0].centroid.x,
                )
                if self.verbose:
                    print(f"📍 Using boundary file: {boundary_path}")
                return boundary_path, (center_y, center_x)
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Could not load boundary file {boundary_path}: {exc}")
                    print("⚠️  Using database centroid for centering")

        center_y, center_x = get_database_centroid(
            self.duckdb_connection, verbose=self.verbose
        )
        return None, (center_y, center_x)

    def _update_ee_boundary(self) -> None:
        if not self.enable_ee:
            self.ee_boundary = None
            return

        if self.effective_boundary_path:
            try:
                boundary_gdf = gpd.read_file(self.effective_boundary_path)
                geometry = shapely.geometry.mapping(boundary_gdf.union_all())
                self.ee_boundary = ee.Geometry(geometry)
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Failed to update Earth Engine boundary: {exc}")
                self.ee_boundary = None
        else:
            self.ee_boundary = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "_owns_connection", False):
            if getattr(self, "duckdb_connection", None):
                self.duckdb_connection.close()
                if self.verbose:
                    print("🔌 DuckDB connection closed.")

    def fetch_embeddings(self, point_ids: List[str], chunk_size: Optional[int] = None):
        if not point_ids:
            return

        chunk_size = chunk_size or DatabaseConstants.EMBEDDING_CHUNK_SIZE
        if len(point_ids) > 100 and self.verbose:
            print(f"🔄 Fetching embeddings for {len(point_ids)} points...")

        for i in range(0, len(point_ids), chunk_size):
            chunk = point_ids[i : i + chunk_size]
            prepared_chunk = prepare_ids_for_query(chunk)
            placeholders = ",".join(["?" for _ in prepared_chunk])
            query = f"""
            SELECT id, tile_id, CAST(embedding AS FLOAT[]) as embedding, geometry 
            FROM geo_embeddings 
            WHERE id IN ({placeholders})
            """
            log_to_file(
                f"Fetch embeddings: Built query for chunk with IDs: {prepared_chunk}"
            )
            arrow_table = self.duckdb_connection.execute(
                query, prepared_chunk
            ).fetch_arrow_table()
            chunk_df = arrow_table.to_pandas()
            yield chunk_df

    def nearest_point(self, lon: float, lat: float):
        sql = DatabaseConstants.NEAREST_POINT_QUERY
        params = [lon, lat]
        return self.duckdb_connection.execute(sql, params).fetchone()

    def query_geometries(self, ids: List[str]):
        if not ids:
            return None
        prepared_ids = prepare_ids_for_query(ids)
        placeholders = ",".join(["?" for _ in prepared_ids])
        sql = f"""
        SELECT ST_AsGeoJSON(geometry) as geometry
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        return self.duckdb_connection.execute(sql, prepared_ids).df()

    def query_search_metadata(self, faiss_ids: List[int]):
        if not faiss_ids:
            return None
        placeholders = ",".join(["?" for _ in faiss_ids])
        sql = f"""
        SELECT id, ST_AsGeoJSON(geometry) AS geometry_json, ST_AsText(geometry) AS geometry_wkt
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        return self.duckdb_connection.execute(sql, faiss_ids).fetchdf()

    def switch_database(self, database_path: str):
        if database_path == self.current_database_path:
            return

        self.current_database_path = database_path
        self.current_database_info = self.database_info_by_path.get(database_path)
        if self.current_database_info:
            self.current_faiss_path = self.current_database_info["faiss_path"]
            self.current_geometry_path = self.current_database_info.get("geometry_path")
        else:
            self.current_faiss_path = None
            self.current_geometry_path = None

        if getattr(self, "_owns_connection", False):
            if getattr(self, "duckdb_connection", None):
                self.duckdb_connection.close()

        self.duckdb_connection = self._connect_duckdb(database_path)
        self._owns_connection = True
        self._apply_duckdb_settings(database_path)

        if not self.current_faiss_path:
            raise RuntimeError(
                f"No FAISS index recorded for {os.path.basename(database_path)}"
            )

        self.faiss_index = faiss.read_index(self.current_faiss_path)
        self.embedding_dim = self._detect_embedding_dim()
        if DatabaseConstants.is_gcs_path(database_path):
            self._warm_up_gcs_database()

        self.effective_boundary_path, (self.center_y, self.center_x) = (
            self._setup_boundary_and_center()
        )
        self._update_ee_boundary()


__all__ = ["DataManager"]
