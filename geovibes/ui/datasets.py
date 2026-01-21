"""Dataset and vector layer management for GeoVibes."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import geopandas as gpd
import numpy as np

from geovibes.ui_config import UIConstants


def detect_geojson_type(geojson_data: Dict) -> str:
    features = geojson_data.get("features", [])
    if not features:
        return "vector_layer"

    first_props = features[0].get("properties", {})

    has_label = "label" in first_props
    has_embedding = "embedding" in first_props
    has_probability = "probability" in first_props

    if has_label and has_embedding:
        return "labeled"
    if has_probability:
        return "detections"
    return "vector_layer"


@dataclass
class DatasetManager:
    data_manager: object
    map_manager: object
    state: object
    verbose: bool = False

    # ------------------------------------------------------------------
    # Dataset save/load
    # ------------------------------------------------------------------

    def save_dataset(self) -> Optional[Dict[str, str]]:
        if not self.state.pos_ids and not self.state.neg_ids:
            if self.verbose:
                print("⚠️ No labeled points to save.")
            return None

        all_ids = list(set(self.state.pos_ids + self.state.neg_ids))
        if not all_ids:
            if self.verbose:
                print("⚠️ No valid labels to save.")
            return None

        prepared_ids = [str(pid) for pid in all_ids]
        placeholders = ",".join(["?" for _ in prepared_ids])

        id_candidates = getattr(self.data_manager, "id_column_candidates", ["id"])
        has_tile_id = "tile_id" in id_candidates
        tile_id_col = "tile_id," if has_tile_id else ""

        query = f"""
        SELECT id,
               {tile_id_col}
               ST_AsGeoJSON(geometry) AS geometry_json,
               embedding
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """

        results = self.data_manager.duckdb_connection.execute(query, prepared_ids).df()
        if results.empty:
            if self.verbose:
                print("⚠️ Could not retrieve data for labeled points.")
            return None

        features = []
        for _, row in results.iterrows():
            point_id = str(row["id"])
            if point_id in self.state.pos_ids:
                label = UIConstants.POSITIVE_LABEL
                class_name = "geovibes_pos"
            elif point_id in self.state.neg_ids:
                label = UIConstants.NEGATIVE_LABEL
                class_name = "geovibes_neg"
            else:
                continue

            if point_id in self.state.cached_embeddings:
                embedding = self.state.cached_embeddings[point_id]
            else:
                embedding = np.array(row["embedding"])

            props = {
                "id": point_id,
                "label": label,
                "class": class_name,
                "embedding": embedding.tolist(),
                "source": "manual",
            }
            if "tile_id" in row.index and row["tile_id"] is not None:
                props["tile_id"] = row["tile_id"]

            features.append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry_json"]),
                    "properties": props,
                }
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_filename = f"labeled_dataset_{timestamp}.geojson"

        geojson_payload = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "positive_points": len(
                    [
                        f
                        for f in features
                        if f["properties"]["label"] == UIConstants.POSITIVE_LABEL
                    ]
                ),
                "negative_points": len(
                    [
                        f
                        for f in features
                        if f["properties"]["label"] == UIConstants.NEGATIVE_LABEL
                    ]
                ),
                "embedding_dimension": getattr(
                    self.data_manager, "embedding_dim", None
                ),
            },
        }

        with open(geojson_filename, "w", encoding="utf-8") as handle:
            json.dump(geojson_payload, handle, indent=2)

        if self.verbose:
            print("✅ Dataset saved successfully!")
            print(f"📄 Filename: {geojson_filename}")

        return {
            "geojson": geojson_filename,
            "positive": str(len(self.state.pos_ids)),
            "negative": str(len(self.state.neg_ids)),
        }

    def export_augmented_dataset(self) -> Optional[Dict[str, str]]:
        if (
            not self.state.pos_ids
            and not self.state.neg_ids
            and not self.state.detection_labels
        ):
            if self.verbose:
                print("⚠️ No labeled points to save.")
            return None

        features = []
        manual_ids = list(set(self.state.pos_ids + self.state.neg_ids))

        id_candidates = getattr(self.data_manager, "id_column_candidates", ["id"])
        has_tile_id = "tile_id" in id_candidates
        tile_id_col = "tile_id," if has_tile_id else ""

        if manual_ids:
            prepared_ids = [str(pid) for pid in manual_ids]
            placeholders = ",".join(["?" for _ in prepared_ids])
            query = f"""
            SELECT id,
                   {tile_id_col}
                   ST_AsGeoJSON(geometry) AS geometry_json,
                   embedding
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
            results = self.data_manager.duckdb_connection.execute(
                query, prepared_ids
            ).df()

            for _, row in results.iterrows():
                point_id = str(row["id"])
                if point_id in self.state.pos_ids:
                    label = UIConstants.POSITIVE_LABEL
                    class_name = "geovibes_pos"
                elif point_id in self.state.neg_ids:
                    label = UIConstants.NEGATIVE_LABEL
                    class_name = "geovibes_neg"
                else:
                    continue

                if point_id in self.state.cached_embeddings:
                    embedding = self.state.cached_embeddings[point_id]
                else:
                    embedding = np.array(row["embedding"])

                props = {
                    "id": point_id,
                    "label": label,
                    "class": class_name,
                    "embedding": embedding.tolist(),
                    "source": "manual",
                }
                if "tile_id" in row.index and row["tile_id"] is not None:
                    props["tile_id"] = row["tile_id"]

                features.append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(row["geometry_json"]),
                        "properties": props,
                    }
                )

        if self.state.detection_labels:
            detection_keys = list(self.state.detection_labels.keys())
            detection_placeholders = ",".join(["?" for _ in detection_keys])

            # Build lookup dict with both string and original keys for robust matching
            labels_lookup = {}
            for k, v in self.state.detection_labels.items():
                labels_lookup[k] = v
                labels_lookup[str(k)] = v

            if has_tile_id:
                detection_query = f"""
                SELECT id,
                       tile_id,
                       ST_AsGeoJSON(geometry) AS geometry_json,
                       embedding
                FROM geo_embeddings
                WHERE tile_id IN ({detection_placeholders})
                """
                id_column_for_lookup = "tile_id"
            else:
                detection_query = f"""
                SELECT id,
                       id AS tile_id,
                       ST_AsGeoJSON(geometry) AS geometry_json,
                       embedding
                FROM geo_embeddings
                WHERE id IN ({detection_placeholders})
                """
                id_column_for_lookup = "id"

            detection_results = self.data_manager.duckdb_connection.execute(
                detection_query, detection_keys
            ).df()

            for _, row in detection_results.iterrows():
                lookup_key = row[id_column_for_lookup]
                detection_label = labels_lookup.get(lookup_key)
                if detection_label is None:
                    detection_label = labels_lookup.get(str(lookup_key))
                if detection_label is None:
                    continue

                if detection_label == 1:
                    label = UIConstants.POSITIVE_LABEL
                    class_name = "relabel_pos"
                else:
                    label = UIConstants.NEGATIVE_LABEL
                    class_name = "relabel_neg"

                embedding = np.array(row["embedding"])
                features.append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(row["geometry_json"]),
                        "properties": {
                            "id": str(row["id"]),
                            "tile_id": str(lookup_key),
                            "label": label,
                            "class": class_name,
                            "embedding": embedding.tolist(),
                            "source": "detection_review",
                        },
                    }
                )

        if not features:
            if self.verbose:
                print("⚠️ No features to export.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_filename = f"augmented_dataset_{timestamp}.geojson"

        manual_positives = len(
            [
                f
                for f in features
                if f["properties"]["source"] == "manual"
                and f["properties"]["label"] == UIConstants.POSITIVE_LABEL
            ]
        )
        manual_negatives = len(
            [
                f
                for f in features
                if f["properties"]["source"] == "manual"
                and f["properties"]["label"] == UIConstants.NEGATIVE_LABEL
            ]
        )
        detection_positives = len(
            [
                f
                for f in features
                if f["properties"]["source"] == "detection_review"
                and f["properties"]["label"] == UIConstants.POSITIVE_LABEL
            ]
        )
        detection_negatives = len(
            [
                f
                for f in features
                if f["properties"]["source"] == "detection_review"
                and f["properties"]["label"] == UIConstants.NEGATIVE_LABEL
            ]
        )

        geojson_payload = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "manual_positives": manual_positives,
                "manual_negatives": manual_negatives,
                "detection_positives": detection_positives,
                "detection_negatives": detection_negatives,
                "embedding_dimension": getattr(
                    self.data_manager, "embedding_dim", None
                ),
            },
        }

        with open(geojson_filename, "w", encoding="utf-8") as handle:
            json.dump(geojson_payload, handle, indent=2)

        if self.verbose:
            print("✅ Augmented dataset saved successfully!")
            print(f"📄 Filename: {geojson_filename}")
            print(f"   Manual: {manual_positives} pos, {manual_negatives} neg")
            print(
                f"   Detection Review: {detection_positives} pos, {detection_negatives} neg"
            )

        return {
            "geojson": geojson_filename,
            "manual_positive": str(manual_positives),
            "manual_negative": str(manual_negatives),
            "detection_positive": str(detection_positives),
            "detection_negative": str(detection_negatives),
        }

    def load_from_content(self, content: bytes, filename: str) -> None:
        if filename.lower().endswith(".geojson"):
            geojson_data = json.loads(content.decode("utf-8"))
            geojson_type = detect_geojson_type(geojson_data)

            if geojson_type == "labeled":
                self._apply_geojson_payload(geojson_data)
            elif geojson_type == "detections":
                self._apply_detection_payload(geojson_data)
            else:
                self.add_vector_from_content(content, filename)
        elif filename.lower().endswith(".parquet"):
            gdf = gpd.read_parquet(io.BytesIO(content))
            self._apply_geodataframe(gdf)
        else:
            raise ValueError(
                "Unsupported file format. Please use .geojson or .parquet files."
            )

    def _apply_geojson_payload(self, payload: Dict) -> None:
        self.state.reset()
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            point_id = str(props.get("id"))
            label = props.get("label")
            embedding = np.array(props.get("embedding", []))
            self.state.cached_embeddings[point_id] = embedding
            if label == UIConstants.POSITIVE_LABEL:
                self.state.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.state.neg_ids.append(point_id)

    def _apply_geodataframe(self, gdf: gpd.GeoDataFrame) -> None:
        required_cols = {"id", "label", "embedding"}
        missing = required_cols - set(gdf.columns)
        if missing:
            raise ValueError(f"Required columns missing: {sorted(missing)}")

        self.state.reset()
        for _, row in gdf.iterrows():
            point_id = str(row["id"])
            embedding = row["embedding"]
            if not isinstance(embedding, (list, np.ndarray)):
                embedding = json.loads(embedding)
            self.state.cached_embeddings[point_id] = np.array(embedding)
            if row["label"] == UIConstants.POSITIVE_LABEL:
                self.state.pos_ids.append(point_id)
            elif row["label"] == UIConstants.NEGATIVE_LABEL:
                self.state.neg_ids.append(point_id)

    def _apply_detection_payload(self, payload: Dict) -> None:
        self.state.detection_mode = True
        self.state.detection_data = payload
        self.map_manager.update_detection_layer(payload)

    # ------------------------------------------------------------------
    # Vector layers
    # ------------------------------------------------------------------

    def add_vector_from_content(self, content: bytes, filename: str) -> None:
        if filename.lower().endswith(".geojson"):
            geojson_data = json.loads(content.decode("utf-8"))
            self.map_manager.set_vector_layer(
                geojson_data, name=f"vector_layer_{filename}"
            )
        elif filename.lower().endswith(".parquet"):
            gdf = gpd.read_parquet(io.BytesIO(content))
            geojson_data = json.loads(gdf.to_json())
            self.map_manager.set_vector_layer(
                geojson_data, name=f"vector_layer_{filename}"
            )
        else:
            raise ValueError(
                "Unsupported file format. Please use .geojson or .parquet files."
            )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def toggle_upload(button, widget, active_label: str, inactive_label: str) -> None:
        if widget.layout.display == "none":
            widget.layout.display = "flex"
            button.description = active_label
        else:
            widget.layout.display = "none"
            button.description = inactive_label
            widget.value = ()

    @staticmethod
    def read_upload_content(payload) -> bytes:
        if isinstance(payload, memoryview):
            return payload.tobytes()
        if isinstance(payload, bytes):
            return payload
        return bytes(payload)


__all__ = ["DatasetManager", "detect_geojson_type"]
