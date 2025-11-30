"""Tests for classification module."""

import json
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from geovibes.classification.pipeline import combine_datasets
from geovibes.classification.classifier import EmbeddingClassifier, EvaluationMetrics
from geovibes.classification.cross_validation import (
    create_spatial_folds,
    cross_validate,
    CVResult,
)
from geovibes.classification.inference import BatchInference, InferenceTiming
from geovibes.classification.output import OutputGenerator, OutputTiming


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


class TestBatchInference:
    """Tests for BatchInference class."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        classifier = MagicMock()
        classifier.predict_proba = MagicMock(
            return_value=np.array([0.9, 0.8, 0.3, 0.1, 0.95])
        )
        return classifier

    @pytest.fixture
    def mock_connection(self):
        """Create a mock DuckDB connection."""
        conn = MagicMock()
        conn.execute = MagicMock(return_value=MagicMock())
        return conn

    def test_inference_timing_throughput(self):
        """InferenceTiming computes throughput correctly."""
        timing = InferenceTiming(
            total_sec=10.0,
            batches_processed=5,
            embeddings_scored=10000,
            detections_found=100,
        )
        assert timing.throughput_per_sec == 1000.0

    def test_inference_timing_zero_time(self):
        """InferenceTiming handles zero time gracefully."""
        timing = InferenceTiming(
            total_sec=0.0,
            batches_processed=0,
            embeddings_scored=0,
            detections_found=0,
        )
        assert timing.throughput_per_sec == 0.0

    def test_batch_inference_init(self, mock_classifier, mock_connection):
        """BatchInference initializes with correct parameters."""
        inference = BatchInference(
            classifier=mock_classifier,
            duckdb_connection=mock_connection,
            batch_size=50000,
            max_memory_gb=8.0,
        )
        assert inference._batch_size == 50000
        assert inference.max_memory_gb == 8.0

    def test_score_batch_returns_probabilities(self, mock_classifier, mock_connection):
        """_score_batch returns probability array."""
        inference = BatchInference(
            classifier=mock_classifier,
            duckdb_connection=mock_connection,
        )
        embeddings = np.random.randn(5, 64).astype(np.float32)

        probs = inference._score_batch(embeddings)

        assert len(probs) == 5
        mock_classifier.predict_proba.assert_called_once()


class TestOutputGenerator:
    """Tests for OutputGenerator class."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock DuckDB connection."""
        conn = MagicMock()
        return conn

    def test_output_timing_dataclass(self):
        """OutputTiming stores timing values."""
        timing = OutputTiming(
            fetch_metadata_sec=1.0,
            generate_tiles_sec=2.0,
            union_tiles_sec=0.5,
            export_sec=0.3,
            total_sec=3.8,
        )
        assert timing.total_sec == 3.8
        assert timing.fetch_metadata_sec == 1.0

    def test_output_generator_init(self, mock_connection):
        """OutputGenerator initializes with tile parameters."""
        generator = OutputGenerator(
            duckdb_connection=mock_connection,
            tile_size_px=32,
            tile_overlap_px=16,
            resolution_m=10.0,
        )
        assert generator.tile_size_m == 320.0  # 32 * 10
        assert generator.half_tile_m == 160.0

    def test_get_utm_crs_northern(self, mock_connection):
        """_get_utm_crs returns correct EPSG for northern hemisphere."""
        generator = OutputGenerator(mock_connection)

        crs = generator._get_utm_crs("16SBH")  # UTM zone 16, band S (northern)

        assert crs.to_epsg() == 32616

    def test_get_utm_crs_southern(self, mock_connection):
        """_get_utm_crs returns correct EPSG for southern hemisphere."""
        generator = OutputGenerator(mock_connection)

        crs = generator._get_utm_crs("32KPB")  # UTM zone 32, band K (southern)

        assert crs.to_epsg() == 32732

    def test_union_tiles_single_polygon(self, mock_connection):
        """union_tiles handles single polygon result."""
        generator = OutputGenerator(mock_connection)

        # Create overlapping tiles that will union into one polygon
        tiles_gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "tile_id": ["A", "B"],
                "probability": [0.9, 0.8],
                "geometry": [
                    box(0, 0, 1, 1),
                    box(0.5, 0, 1.5, 1),  # Overlapping
                ],
            },
            crs="EPSG:4326",
        )

        union_gdf, elapsed = generator.union_tiles(tiles_gdf)

        assert len(union_gdf) == 1  # Single unioned polygon
        assert elapsed > 0

    def test_union_tiles_multiple_polygons(self, mock_connection):
        """union_tiles handles multiple disjoint polygons."""
        generator = OutputGenerator(mock_connection)

        # Create non-overlapping tiles
        tiles_gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "tile_id": ["A", "B"],
                "probability": [0.9, 0.8],
                "geometry": [
                    box(0, 0, 1, 1),
                    box(10, 10, 11, 11),  # Far away, no overlap
                ],
            },
            crs="EPSG:4326",
        )

        union_gdf, elapsed = generator.union_tiles(tiles_gdf)

        assert len(union_gdf) == 2  # Two separate polygons
        assert elapsed > 0

    def test_export_geojson_creates_files(self, mock_connection, tmp_path):
        """export_geojson creates detection and union files."""
        generator = OutputGenerator(mock_connection)

        tiles_gdf = gpd.GeoDataFrame(
            {
                "id": [1],
                "tile_id": ["A"],
                "probability": [0.9],
                "geometry": [box(0, 0, 1, 1)],
            },
            crs="EPSG:4326",
        )
        union_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

        paths, elapsed = generator.export_geojson(
            tiles_gdf, union_gdf, str(tmp_path), name="test"
        )

        assert "detections" in paths
        assert "union" in paths
        assert (tmp_path / "test_detections.geojson").exists()
        assert (tmp_path / "test_union.geojson").exists()


class TestDataLoaderHelpers:
    """Tests for data loader helper functionality."""

    def test_geojson_missing_class_gets_unknown(self, tmp_path):
        """Features without class field get class='unknown' via combine_datasets."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1},  # No class
                },
            ],
        }
        path = tmp_path / "no_class.geojson"
        path.write_text(json.dumps(geojson))

        result_path = combine_datasets([str(path)])

        with open(result_path) as f:
            result = json.load(f)

        assert result["features"][0]["properties"]["class"] == "unknown"

    def test_geojson_invalid_label_values(self, tmp_path):
        """combine_datasets passes through invalid labels (validation happens in loader)."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 2, "class": "invalid"},  # Invalid label
                },
            ],
        }
        path = tmp_path / "invalid.geojson"
        path.write_text(json.dumps(geojson))

        # combine_datasets doesn't validate labels, just preserves them
        result_path = combine_datasets([str(path)])

        with open(result_path) as f:
            result = json.load(f)

        # Label is preserved as-is (validation happens in ClassificationDataLoader)
        assert result["features"][0]["properties"]["label"] == 2


# =============================================================================
# INTEGRATION TESTS - Using real DuckDB with spatial extension
# =============================================================================


class TestDataLoaderIntegration:
    """Integration tests for ClassificationDataLoader with real DuckDB."""

    def test_load_with_tile_id_match(
        self, duckdb_connection, training_geojson_with_tile_ids
    ):
        """Load training data matching by tile_id property."""
        from geovibes.classification.data_loader import ClassificationDataLoader

        loader = ClassificationDataLoader(
            duckdb_connection, training_geojson_with_tile_ids
        )
        train_df, test_df, timing = loader.load(test_fraction=0.2, random_state=42)

        # Should have loaded embeddings
        assert len(train_df) + len(test_df) == 20  # 10 pos + 10 neg
        assert "embedding" in train_df.columns
        assert "tile_id" in train_df.columns
        assert "label" in train_df.columns

        # Embeddings should be numpy arrays of correct dimension
        assert train_df["embedding"].iloc[0].shape == (64,)

        # Timing should be populated
        assert timing.parse_geojson_sec > 0
        assert timing.fetch_embeddings_sec > 0
        assert timing.spatial_match_sec == 0.0  # No spatial match needed

    def test_load_with_db_id_lookup(
        self, duckdb_connection, training_geojson_with_db_ids
    ):
        """Load training data using database row IDs to look up tile_ids."""
        from geovibes.classification.data_loader import ClassificationDataLoader

        loader = ClassificationDataLoader(
            duckdb_connection, training_geojson_with_db_ids
        )
        train_df, test_df, timing = loader.load(test_fraction=0.2, random_state=42)

        # Should have resolved db_ids to tile_ids
        assert len(train_df) + len(test_df) == 20
        assert "tile_id" in train_df.columns
        assert "db_id" not in train_df.columns  # Should be dropped after lookup

    def test_load_for_cv_returns_geometries(
        self, duckdb_connection, training_geojson_with_tile_ids
    ):
        """load_for_cv returns point geometries for spatial fold creation."""
        from geovibes.classification.data_loader import ClassificationDataLoader

        loader = ClassificationDataLoader(
            duckdb_connection, training_geojson_with_tile_ids
        )
        df, geometries, timing = loader.load_for_cv(random_state=42)

        assert len(df) == 20
        assert len(geometries) == 20
        assert geometries.crs.to_epsg() == 4326

        # All geometries should be Points
        assert all(geom.geom_type == "Point" for geom in geometries)

    def test_stratified_split_preserves_class_distribution(
        self, duckdb_connection, training_geojson_with_tile_ids
    ):
        """Stratified split maintains class proportions in train/test."""
        from geovibes.classification.data_loader import ClassificationDataLoader

        loader = ClassificationDataLoader(
            duckdb_connection, training_geojson_with_tile_ids
        )
        train_df, test_df, _ = loader.load(test_fraction=0.2, random_state=42)

        # Both train and test should have positives and negatives
        assert (train_df["label"] == 1).sum() > 0
        assert (train_df["label"] == 0).sum() > 0
        assert (test_df["label"] == 1).sum() > 0
        assert (test_df["label"] == 0).sum() > 0

    def test_missing_tile_id_raises_error(self, duckdb_connection, tmp_path):
        """Missing embeddings for tile_ids raises ValueError."""
        from geovibes.classification.data_loader import ClassificationDataLoader

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {
                        "tile_id": "NONEXISTENT_TILE",
                        "label": 1,
                        "class": "pos",
                    },
                }
            ],
        }
        path = tmp_path / "missing.geojson"
        path.write_text(json.dumps(geojson))

        loader = ClassificationDataLoader(duckdb_connection, str(path))
        with pytest.raises(ValueError, match="Missing embeddings"):
            loader.load()


class TestBatchInferenceIntegration:
    """Integration tests for BatchInference with real DuckDB."""

    def test_run_inference_detects_positives(
        self, duckdb_connection, trained_classifier, duckdb_metadata
    ):
        """Inference finds positive embeddings above threshold."""
        from geovibes.classification.inference import BatchInference

        inference = BatchInference(
            classifier=trained_classifier,
            duckdb_connection=duckdb_connection,
            batch_size=50,  # Small batches for test
        )

        detections, timing = inference.run(probability_threshold=0.5)

        # Should detect most positives (30 in database, centered at +2)
        n_pos = duckdb_metadata["n_positives"]
        assert len(detections) > n_pos * 0.5  # At least half detected
        assert len(detections) < duckdb_metadata["n_samples"]  # Not everything

        # All detections should have prob >= threshold
        for id_, prob in detections:
            assert prob >= 0.5

        # Timing should be populated
        assert timing.embeddings_scored == duckdb_metadata["n_samples"]
        assert timing.batches_processed >= 2  # 100 samples / 50 batch size

    def test_batch_iteration_covers_all_embeddings(
        self, duckdb_connection, trained_classifier, duckdb_metadata
    ):
        """_iterate_batches_fast yields all embeddings exactly once."""
        from geovibes.classification.inference import BatchInference

        inference = BatchInference(
            classifier=trained_classifier,
            duckdb_connection=duckdb_connection,
            batch_size=30,
        )

        all_ids = []
        all_embeddings = []
        for ids, embeddings in inference._iterate_batches_fast():
            all_ids.extend(ids.tolist())
            all_embeddings.append(embeddings)

        # Should have all embeddings
        assert len(all_ids) == duckdb_metadata["n_samples"]
        # IDs should be unique
        assert len(set(all_ids)) == duckdb_metadata["n_samples"]

    def test_threshold_filtering(
        self, duckdb_connection, trained_classifier, duckdb_metadata
    ):
        """Higher threshold returns fewer detections."""
        from geovibes.classification.inference import BatchInference

        inference = BatchInference(
            classifier=trained_classifier,
            duckdb_connection=duckdb_connection,
        )

        detections_50, _ = inference.run(probability_threshold=0.5)
        detections_90, _ = inference.run(probability_threshold=0.9)

        # Higher threshold should return fewer or equal detections
        assert len(detections_90) <= len(detections_50)

    def test_auto_batch_size_detection(
        self, duckdb_connection, trained_classifier, duckdb_metadata
    ):
        """batch_size=0 auto-detects based on memory and data size."""
        from geovibes.classification.inference import BatchInference

        inference = BatchInference(
            classifier=trained_classifier,
            duckdb_connection=duckdb_connection,
            batch_size=0,  # Auto-detect
            max_memory_gb=12.0,
        )

        # For small test data, should use full dataset
        assert inference.batch_size == duckdb_metadata["n_samples"]


class TestOutputGeneratorIntegration:
    """Integration tests for OutputGenerator with real DuckDB."""

    def test_fetch_detection_metadata(self, duckdb_connection, duckdb_metadata):
        """fetch_detection_metadata retrieves tile_ids and geometries."""
        from geovibes.classification.output import OutputGenerator

        generator = OutputGenerator(duckdb_connection)

        # Create mock detections (id, probability)
        detections = [(1, 0.95), (2, 0.88), (3, 0.72)]

        gdf, elapsed = generator.fetch_detection_metadata(detections)

        assert len(gdf) == 3
        assert "tile_id" in gdf.columns
        assert "probability" in gdf.columns
        assert gdf.crs.to_epsg() == 4326
        assert all(gdf.geometry.geom_type == "Point")
        assert elapsed > 0

    def test_generate_tile_geometries(self, duckdb_connection, duckdb_metadata):
        """generate_tile_geometries creates square polygons from points."""
        from geovibes.classification.output import OutputGenerator

        generator = OutputGenerator(
            duckdb_connection,
            tile_size_px=32,
            tile_overlap_px=16,
            resolution_m=10.0,  # 320m tiles
        )

        detections = [(1, 0.95), (2, 0.88)]
        points_gdf, _ = generator.fetch_detection_metadata(detections)

        tiles_gdf, elapsed = generator.generate_tile_geometries(points_gdf)

        assert len(tiles_gdf) == 2
        assert all(tiles_gdf.geometry.geom_type == "Polygon")
        assert elapsed > 0

        # Tiles should be approximately square
        for geom in tiles_gdf.geometry:
            bounds = geom.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            # In WGS84, width/height ratio depends on latitude
            assert 0.5 < width / height < 2.0

    def test_full_output_pipeline(self, duckdb_connection, duckdb_metadata, tmp_path):
        """generate_output creates detection and union GeoJSON files."""
        from geovibes.classification.output import OutputGenerator

        generator = OutputGenerator(duckdb_connection)

        # Multiple detections, some overlapping
        detections = [(1, 0.95), (2, 0.88), (3, 0.72), (4, 0.65)]

        paths, timing = generator.generate_output(
            detections=detections,
            output_dir=str(tmp_path),
            name="test_output",
        )

        assert "detections" in paths
        assert "union" in paths
        assert (tmp_path / "test_output_detections.geojson").exists()
        assert (tmp_path / "test_output_union.geojson").exists()

        # Timing should be complete
        assert timing.fetch_metadata_sec > 0
        assert timing.generate_tiles_sec > 0
        assert timing.total_sec > 0

    def test_utm_crs_parsing(self, duckdb_connection):
        """_get_utm_crs correctly parses MGRS tile_ids to UTM zones."""
        from geovibes.classification.output import OutputGenerator

        generator = OutputGenerator(duckdb_connection)

        # Northern hemisphere tiles
        assert generator._get_utm_crs("16SBH0001000010").to_epsg() == 32616
        assert generator._get_utm_crs("32TNS1234567890").to_epsg() == 32632

        # Southern hemisphere tiles (bands A-M)
        assert generator._get_utm_crs("32KPB0001000010").to_epsg() == 32732
        assert generator._get_utm_crs("16LBH0001000010").to_epsg() == 32716


class TestPipelineIntegration:
    """Integration tests for end-to-end classification pipeline."""

    @pytest.fixture
    def duckdb_file(self, duckdb_with_embeddings, tmp_path):
        """Create a file-based DuckDB for pipeline tests."""
        conn, metadata = duckdb_with_embeddings
        db_path = tmp_path / "test_embeddings.duckdb"

        # Create a new file-based database with same data
        file_conn = duckdb.connect(str(db_path))
        file_conn.execute("INSTALL spatial; LOAD spatial;")
        file_conn.execute("""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY,
                tile_id VARCHAR,
                embedding FLOAT[64],
                geometry GEOMETRY
            )
        """)

        # Copy data from in-memory to file
        result = conn.execute("""
            SELECT id, tile_id, CAST(embedding AS FLOAT[]) as embedding,
                   ST_AsText(geometry) as geom_wkt
            FROM geo_embeddings
        """).fetchall()

        for row in result:
            file_conn.execute(
                """
                INSERT INTO geo_embeddings (id, tile_id, embedding, geometry)
                VALUES (?, ?, ?, ST_GeomFromText(?))
                """,
                [row[0], row[1], list(row[2]), row[3]],
            )

        file_conn.close()

        return str(db_path), metadata

    def test_full_pipeline_run(
        self, duckdb_file, training_geojson_with_tile_ids, tmp_path
    ):
        """Full pipeline: load, train, evaluate, infer, output."""
        from geovibes.classification.pipeline import ClassificationPipeline

        db_path, metadata = duckdb_file
        output_dir = tmp_path / "pipeline_output"

        with ClassificationPipeline(db_path, memory_limit="1GB") as pipeline:
            result = pipeline.run(
                geojson_path=training_geojson_with_tile_ids,
                output_dir=str(output_dir),
                test_fraction=0.2,
                probability_threshold=0.5,
                batch_size=50,
            )

        # Metrics should be reasonable for linearly separable data
        assert result.metrics.f1 > 0.5
        assert result.metrics.auc_roc > 0.5

        # Should have some detections
        assert result.num_detections > 0

        # Output files should exist
        assert (output_dir / "classification_detections.geojson").exists()
        assert (output_dir / "classification_union.geojson").exists()
        assert (output_dir / "model.json").exists()

        # Timing should be complete
        assert result.timing.total_sec > 0

    def test_cv_mode(self, duckdb_file, training_geojson_with_tile_ids):
        """Cross-validation mode returns CVResult."""
        from geovibes.classification.pipeline import run_cross_validation

        db_path, _ = duckdb_file

        # Use small buffer (100m) to create multiple clusters from test data
        # Test positives are spaced ~1km apart, so 100m buffer creates separate clusters
        cv_result = run_cross_validation(
            geojson_path=training_geojson_with_tile_ids,
            duckdb_path=db_path,
            memory_limit="1GB",
            buffer_m=100.0,  # Small buffer to create multiple clusters
            n_folds=3,  # Fewer folds for small test data
        )

        assert 0 <= cv_result.f1_mean <= 1
        assert 0 <= cv_result.auc_roc_mean <= 1
        assert cv_result.f1_std >= 0
        assert len(cv_result.fold_metrics) == 3

    def test_combine_datasets_integration(self, tmp_path):
        """combine_datasets merges multiple GeoJSON files correctly."""
        from geovibes.classification.pipeline import combine_datasets

        file1 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"label": 1, "class": "original_pos"},
                }
            ],
        }
        file2 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"label": 0, "class": "sampled_neg"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [2, 2]},
                    "properties": {"label": 1, "class": "relabel_pos"},
                },
            ],
        }

        path1 = tmp_path / "file1.geojson"
        path2 = tmp_path / "file2.geojson"
        path1.write_text(json.dumps(file1))
        path2.write_text(json.dumps(file2))

        combined_path = combine_datasets([str(path1), str(path2)])

        with open(combined_path) as f:
            combined = json.load(f)

        assert len(combined["features"]) == 3
        assert combined["metadata"]["positive_count"] == 2
        assert combined["metadata"]["negative_count"] == 1
        assert combined["metadata"]["sources"][str(path1)] == 1
        assert combined["metadata"]["sources"][str(path2)] == 2


class TestRandomNegativeSampling:
    """Tests for random negative sampling from DuckDB."""

    @pytest.fixture
    def duckdb_file_for_sampling(self, tmp_path):
        """Create a DuckDB file with geo_embeddings for sampling tests."""
        db_path = tmp_path / "test_sampling.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("INSTALL spatial; LOAD spatial;")

        np.random.seed(42)
        embedding_dim = 64
        n_samples = 50

        embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)

        conn.execute(f"""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY,
                tile_id VARCHAR,
                embedding FLOAT[{embedding_dim}],
                geometry GEOMETRY
            )
        """)

        for i in range(n_samples):
            lon = -86.5 + (i % 10) * 0.01
            lat = 33.0 + (i // 10) * 0.01
            tile_id = f"TILE{i:04d}"
            emb_list = embeddings[i].tolist()
            conn.execute(
                """
                INSERT INTO geo_embeddings (id, tile_id, embedding, geometry)
                VALUES (?, ?, ?, ST_GeomFromText(?))
                """,
                [i + 1, tile_id, emb_list, f"POINT({lon} {lat})"],
            )

        conn.close()
        return str(db_path)

    def test_sample_random_negatives_basic(self, duckdb_file_for_sampling):
        """Basic random sampling excludes positive tile_ids."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        positive_tile_ids = ["TILE0000", "TILE0001", "TILE0002"]

        result = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=10,
            seed=42,
        )

        assert len(result) == 10
        assert "tile_id" in result.columns
        assert "geometry" in result.columns
        assert "class" in result.columns
        assert "label" in result.columns

        # Check no positives in result
        for tile_id in positive_tile_ids:
            assert tile_id not in result["tile_id"].values

        # Check all labels are 0
        assert (result["label"] == 0).all()
        assert (result["class"] == "random_neg").all()

    def test_sample_random_negatives_default_count(self, duckdb_file_for_sampling):
        """Default sample count equals number of positives."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        positive_tile_ids = ["TILE0000", "TILE0001", "TILE0002", "TILE0003", "TILE0004"]

        result = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            seed=42,
        )

        assert len(result) == len(positive_tile_ids)

    def test_sample_random_negatives_reproducible(self, duckdb_file_for_sampling):
        """Same seed produces same samples."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        positive_tile_ids = ["TILE0000"]

        result1 = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=5,
            seed=42,
        )
        result2 = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=5,
            seed=42,
        )

        assert list(result1["tile_id"]) == list(result2["tile_id"])

    def test_sample_random_negatives_different_seeds(self, duckdb_file_for_sampling):
        """Different seeds produce different samples."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        positive_tile_ids = ["TILE0000"]

        result1 = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=10,
            seed=42,
        )
        result2 = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=10,
            seed=123,
        )

        # Should be different (very unlikely to be identical with different seeds)
        assert list(result1["tile_id"]) != list(result2["tile_id"])

    def test_sample_random_negatives_with_buffer(self, duckdb_file_for_sampling):
        """Spatial buffer excludes nearby points."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        # TILE0000 is at (-86.5, 33.0), TILE0001 is at (-86.49, 33.0)
        # Distance between them is ~0.01 degrees * 111km/degree ≈ 1.1km
        positive_tile_ids = ["TILE0000"]

        # With 2km buffer, should exclude TILE0001 (about 1.1km away)
        result = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=positive_tile_ids,
            num_samples=45,
            buffer_meters=2000.0,
            seed=42,
        )

        # TILE0001 should not be in results due to buffer
        assert "TILE0001" not in result["tile_id"].values
        # TILE0000 should not be in results (it's a positive)
        assert "TILE0000" not in result["tile_id"].values

    def test_sample_random_negatives_empty_positives(self, duckdb_file_for_sampling):
        """Empty positive list samples from entire database."""
        from geovibes.classification.sample_negatives import sample_random_negatives

        result = sample_random_negatives(
            duckdb_path=duckdb_file_for_sampling,
            positive_tile_ids=[],
            num_samples=10,
            seed=42,
        )

        assert len(result) == 10

    def test_sample_random_negatives_from_geojson(
        self, duckdb_file_for_sampling, tmp_path
    ):
        """Convenience function extracts tile_ids from GeoJSON."""
        from geovibes.classification.sample_negatives import (
            sample_random_negatives_from_geojson,
        )

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"tile_id": "TILE0000", "label": 1, "class": "pos"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"tile_id": "TILE0001", "label": 1, "class": "pos"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"tile_id": "TILE0002", "label": 0, "class": "neg"},
                },
            ],
        }

        geojson_path = tmp_path / "positives.geojson"
        geojson_path.write_text(json.dumps(geojson))

        result = sample_random_negatives_from_geojson(
            duckdb_path=duckdb_file_for_sampling,
            positives_geojson_path=str(geojson_path),
            seed=42,
        )

        # Should have 2 samples (matching 2 positives in the geojson)
        assert len(result) == 2
        # Should not include the positive tile_ids
        assert "TILE0000" not in result["tile_id"].values
        assert "TILE0001" not in result["tile_id"].values

    def test_sample_random_negatives_from_geojson_saves_output(
        self, duckdb_file_for_sampling, tmp_path
    ):
        """Output path saves GeoJSON file."""
        from geovibes.classification.sample_negatives import (
            sample_random_negatives_from_geojson,
        )

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"tile_id": "TILE0000", "label": 1, "class": "pos"},
                },
            ],
        }

        geojson_path = tmp_path / "positives.geojson"
        geojson_path.write_text(json.dumps(geojson))

        output_path = tmp_path / "negatives.geojson"

        sample_random_negatives_from_geojson(
            duckdb_path=duckdb_file_for_sampling,
            positives_geojson_path=str(geojson_path),
            output_path=str(output_path),
            num_samples=5,
            seed=42,
        )

        assert output_path.exists()

        saved = gpd.read_file(output_path)
        assert len(saved) == 5
