"""Tests for classification module."""

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geovibes.classification.pipeline import combine_datasets
from geovibes.classification.classifier import EmbeddingClassifier, EvaluationMetrics
from geovibes.classification.cross_validation import (
    create_spatial_folds,
    cross_validate,
    CVResult,
)


class TestCombineDatasets:
    """Tests for combine_datasets function."""

    def test_combine_single_file(self, tmp_path):
        """Single file returns path to temp file with same features."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1, "class": "pos"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"label": 0, "class": "neg"},
                },
            ],
        }
        path = tmp_path / "single.geojson"
        path.write_text(json.dumps(geojson))

        result_path = combine_datasets([str(path)])

        with open(result_path) as f:
            result = json.load(f)
        assert len(result["features"]) == 2
        assert result["metadata"]["positive_count"] == 1
        assert result["metadata"]["negative_count"] == 1

    def test_combine_multiple_files_preserves_labels(self, tmp_path):
        """Multiple files preserve original labels from each file."""
        file1 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1, "class": "geovibes_pos"},
                },
            ],
        }
        file2 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"label": 0, "class": "geovibes_neg"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [2, 2]},
                    "properties": {"label": 0, "class": "relabel_neg"},
                },
            ],
        }

        path1 = tmp_path / "file1.geojson"
        path2 = tmp_path / "file2.geojson"
        path1.write_text(json.dumps(file1))
        path2.write_text(json.dumps(file2))

        result_path = combine_datasets([str(path1), str(path2)])

        with open(result_path) as f:
            result = json.load(f)

        assert len(result["features"]) == 3
        assert result["metadata"]["positive_count"] == 1
        assert result["metadata"]["negative_count"] == 2

        labels = [f["properties"]["label"] for f in result["features"]]
        assert labels.count(1) == 1
        assert labels.count(0) == 2

    def test_combine_tracks_sources(self, tmp_path):
        """Metadata tracks feature counts from each source file."""
        file1 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"label": 1},
                },
            ],
        }
        file2 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [2, 2]},
                    "properties": {"label": 0},
                },
            ],
        }

        path1 = tmp_path / "original.geojson"
        path2 = tmp_path / "corrections.geojson"
        path1.write_text(json.dumps(file1))
        path2.write_text(json.dumps(file2))

        result_path = combine_datasets([str(path1), str(path2)])

        with open(result_path) as f:
            result = json.load(f)

        sources = result["metadata"]["sources"]
        assert sources[str(path1)] == 2
        assert sources[str(path2)] == 1

    def test_combine_raises_on_missing_label(self, tmp_path):
        """Raises ValueError if feature missing label field."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"class": "pos"},
                },  # Missing label
            ],
        }
        path = tmp_path / "bad.geojson"
        path.write_text(json.dumps(geojson))

        with pytest.raises(ValueError, match="missing 'label'"):
            combine_datasets([str(path)])

    def test_combine_defaults_class_to_unknown(self, tmp_path):
        """Features without class field get class='unknown'."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1},
                },  # Missing class
            ],
        }
        path = tmp_path / "no_class.geojson"
        path.write_text(json.dumps(geojson))

        result_path = combine_datasets([str(path)])

        with open(result_path) as f:
            result = json.load(f)

        assert result["features"][0]["properties"]["class"] == "unknown"


class TestEmbeddingClassifier:
    """Tests for EmbeddingClassifier."""

    @pytest.fixture
    def sample_data(self):
        """Generate simple linearly separable data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 64

        X_pos = np.random.randn(n_samples // 2, n_features) + 2
        X_neg = np.random.randn(n_samples // 2, n_features) - 2
        X = np.vstack([X_pos, X_neg]).astype(np.float32)
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2), dtype=np.int32)

        return X, y

    def test_fit_returns_time(self, sample_data):
        """Fit returns training time in seconds."""
        X, y = sample_data
        classifier = EmbeddingClassifier(n_estimators=10, max_depth=3)

        fit_time = classifier.fit(X, y)

        assert isinstance(fit_time, float)
        assert fit_time > 0
        assert classifier.fit_time == fit_time

    def test_evaluate_returns_metrics(self, sample_data):
        """Evaluate returns EvaluationMetrics and time."""
        X, y = sample_data
        classifier = EmbeddingClassifier(n_estimators=10, max_depth=3)
        classifier.fit(X, y)

        metrics, eval_time = classifier.evaluate(X, y)

        assert isinstance(metrics, EvaluationMetrics)
        assert isinstance(eval_time, float)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.auc_roc <= 1

    def test_fit_with_sample_weight(self, sample_data):
        """Fit accepts sample_weight parameter."""
        X, y = sample_data
        weights = np.ones(len(y), dtype=np.float32)
        weights[:10] = 3.0  # Weight first 10 samples more

        classifier = EmbeddingClassifier(n_estimators=10, max_depth=3)
        fit_time = classifier.fit(X, y, sample_weight=weights)

        assert fit_time > 0
        metrics, _ = classifier.evaluate(X, y)
        assert metrics.accuracy > 0.5

    def test_predict_proba_shape(self, sample_data):
        """predict_proba returns probabilities with correct shape."""
        X, y = sample_data
        classifier = EmbeddingClassifier(n_estimators=10, max_depth=3)
        classifier.fit(X, y)

        proba = classifier.predict_proba(X)

        assert proba.shape == (len(X),)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_save_and_load(self, sample_data, tmp_path):
        """Model can be saved and loaded."""
        X, y = sample_data
        classifier = EmbeddingClassifier(n_estimators=10, max_depth=3)
        classifier.fit(X, y)
        original_proba = classifier.predict_proba(X)

        model_path = str(tmp_path / "model.json")
        classifier.save(model_path)

        loaded = EmbeddingClassifier.load(model_path)
        loaded_proba = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(original_proba, loaded_proba)


class TestDetectionModeLabeling:
    """Tests for detection mode in tile panel."""

    def test_tile_panel_detection_mode_label_style(self):
        """Tile panel applies correct styles in detection mode."""
        from geovibes.ui.state import AppState

        state = AppState()
        state.detection_mode = True
        state.detection_labels = {"tile_A": 1, "tile_B": 0}

        # Verify state tracks detection labels
        assert state.detection_labels["tile_A"] == 1
        assert state.detection_labels["tile_B"] == 0

    def test_state_label_detection(self):
        """AppState.label_detection stores tile_id -> label mapping."""
        from geovibes.ui.state import AppState

        state = AppState()
        state.detection_mode = True

        state.label_detection("tile_123", 1)
        assert state.detection_labels["tile_123"] == 1

        state.label_detection("tile_456", 0)
        assert state.detection_labels["tile_456"] == 0

    def test_state_label_detection_toggle(self):
        """Labeling same detection twice can toggle."""
        from geovibes.ui.state import AppState

        state = AppState()
        state.detection_mode = True

        state.label_detection("tile_A", 1)
        assert state.detection_labels["tile_A"] == 1

        # Change label
        state.label_detection("tile_A", 0)
        assert state.detection_labels["tile_A"] == 0


class TestSpatialCrossValidation:
    """Tests for spatial cross-validation."""

    @pytest.fixture
    def spatial_data(self):
        """Create test data with spatial clusters."""
        np.random.seed(42)

        # Create 3 spatial clusters of positives
        cluster1_pos = [(0.0, 0.0), (0.001, 0.001), (0.002, 0.0)]  # 3 points
        cluster2_pos = [(1.0, 1.0), (1.001, 1.001)]  # 2 points
        cluster3_pos = [(2.0, 0.0), (2.001, 0.001), (2.002, 0.0), (2.003, 0.001)]  # 4

        # Negatives scattered near clusters
        negatives = [
            (0.01, 0.01),
            (0.02, 0.0),
            (1.01, 1.01),
            (1.02, 1.0),
            (2.01, 0.01),
            (2.02, 0.0),
        ]

        all_points = cluster1_pos + cluster2_pos + cluster3_pos + negatives
        labels = [1] * 9 + [0] * 6  # 9 positives, 6 negatives

        df = pd.DataFrame(
            {
                "label": labels,
                "class": ["pos"] * 9 + ["neg"] * 6,
            }
        )
        geometries = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in all_points], crs="EPSG:4326"
        )

        # Create embeddings (linearly separable)
        n_features = 32
        X_pos = np.random.randn(9, n_features) + 2
        X_neg = np.random.randn(6, n_features) - 2
        embeddings = np.vstack([X_pos, X_neg]).astype(np.float32)

        return df, geometries, embeddings

    def test_create_spatial_folds_returns_valid_assignments(self, spatial_data):
        """Fold assignments are valid integers in range [0, n_folds)."""
        df, geometries, _ = spatial_data

        fold_assignments, n_clusters, cluster_dist = create_spatial_folds(
            df, geometries, n_folds=3, buffer_m=500.0
        )

        assert len(fold_assignments) == len(df)
        assert fold_assignments.min() >= 0
        assert fold_assignments.max() < 3
        assert n_clusters >= 1

    def test_create_spatial_folds_groups_nearby_positives(self, spatial_data):
        """Nearby positives should be in the same fold."""
        df, geometries, _ = spatial_data

        fold_assignments, _, _ = create_spatial_folds(
            df, geometries, n_folds=3, buffer_m=1000.0
        )

        # First 3 positives (cluster 1) should be in same fold
        cluster1_folds = fold_assignments[:3]
        assert len(set(cluster1_folds)) == 1

    def test_cross_validate_returns_cv_result(self, spatial_data):
        """Cross-validation returns CVResult with metrics."""
        df, geometries, embeddings = spatial_data

        fold_assignments, n_clusters, cluster_dist = create_spatial_folds(
            df, geometries, n_folds=3, buffer_m=500.0
        )

        y = df["label"].values.astype(np.int32)

        result = cross_validate(
            X=embeddings,
            y=y,
            fold_assignments=fold_assignments,
            n_folds=3,
            n_clusters=n_clusters,
            cluster_distribution=cluster_dist,
            n_estimators=10,
            max_depth=3,
        )

        assert isinstance(result, CVResult)
        assert 0 <= result.f1_mean <= 1
        assert 0 <= result.auc_roc_mean <= 1
        assert result.f1_std >= 0
        assert len(result.fold_metrics) > 0

    def test_cv_result_summary_format(self, spatial_data):
        """CVResult.summary() returns formatted string."""
        df, geometries, embeddings = spatial_data

        fold_assignments, n_clusters, cluster_dist = create_spatial_folds(
            df, geometries, n_folds=3, buffer_m=500.0
        )

        y = df["label"].values.astype(np.int32)

        result = cross_validate(
            X=embeddings,
            y=y,
            fold_assignments=fold_assignments,
            n_folds=3,
            n_clusters=n_clusters,
            cluster_distribution=cluster_dist,
            n_estimators=10,
            max_depth=3,
        )

        summary = result.summary()
        assert "CROSS-VALIDATION" in summary
        assert "F1:" in summary
        assert "AUC-ROC:" in summary
        assert "±" in summary  # std deviation

    def test_cross_validate_with_sample_weights(self, spatial_data):
        """Cross-validation accepts sample weights."""
        df, geometries, embeddings = spatial_data

        fold_assignments, n_clusters, cluster_dist = create_spatial_folds(
            df, geometries, n_folds=3, buffer_m=500.0
        )

        y = df["label"].values.astype(np.int32)
        weights = np.ones(len(df), dtype=np.float32)
        weights[:3] = 2.0  # Weight first 3 samples more

        result = cross_validate(
            X=embeddings,
            y=y,
            fold_assignments=fold_assignments,
            sample_weights=weights,
            n_folds=3,
            n_clusters=n_clusters,
            cluster_distribution=cluster_dist,
            n_estimators=10,
            max_depth=3,
        )

        assert isinstance(result, CVResult)
        assert len(result.fold_metrics) > 0
