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
        query = f"""
        SELECT id,
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
            elif point_id in self.state.neg_ids:
                label = UIConstants.NEGATIVE_LABEL
            else:
                continue

            if point_id in self.state.cached_embeddings:
                embedding = self.state.cached_embeddings[point_id]
            else:
                embedding = np.array(row["embedding"])

            features.append(
                {
                    "type": "Feature",
                    "geometry": json.loads(row["geometry_json"]),
                    "properties": {
                        "id": point_id,
                        "label": label,
                        "embedding": embedding.tolist(),
                    },
                }
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_filename = f"labeled_dataset_{timestamp}.geojson"
        csv_filename = f"labeled_dataset_{timestamp}_labels.csv"

        geojson_payload = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "positive_points": len(
                    [f for f in features if f["properties"]["label"] == UIConstants.POSITIVE_LABEL]
                ),
                "negative_points": len(
                    [f for f in features if f["properties"]["label"] == UIConstants.NEGATIVE_LABEL]
                ),
                "embedding_dimension": getattr(self.data_manager, "embedding_dim", None),
            },
        }

        with open(geojson_filename, "w", encoding="utf-8") as handle:
            json.dump(geojson_payload, handle, indent=2)

        import pandas as pd

        labels_df = pd.DataFrame(
            {"id": f["properties"]["id"], "label": f["properties"]["label"]}
            for f in features
        )
        labels_df.to_csv(csv_filename, index=False)

        if self.verbose:
            print("✅ Dataset saved successfully!")
            print(f"📄 Filename: {geojson_filename}")
            print(f"📄 Labels CSV: {csv_filename}")

        return {
            "geojson": geojson_filename,
            "csv": csv_filename,
            "positive": str(len(self.state.pos_ids)),
            "negative": str(len(self.state.neg_ids)),
        }

    def load_from_content(self, content: bytes, filename: str) -> None:
        if filename.lower().endswith(".geojson"):
            geojson_data = json.loads(content.decode("utf-8"))
            self._apply_geojson_payload(geojson_data)
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


__all__ = ["DatasetManager"]
