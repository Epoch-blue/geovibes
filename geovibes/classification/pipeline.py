"""
Classification Pipeline

Orchestrates the full classification workflow:
1. Load and split training data
2. Train XGBoost classifier
3. Evaluate on test set
4. Run inference on all embeddings
5. Generate output GeoJSON

All steps are timed and reported.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import numpy as np
import time
import os
import duckdb

from geovibes.classification.data_loader import ClassificationDataLoader, LoaderTiming
from geovibes.classification.classifier import (
    EmbeddingClassifier,
    EvaluationMetrics,
)
from geovibes.classification.inference import BatchInference, InferenceTiming
from geovibes.classification.output import OutputGenerator, OutputTiming


@dataclass
class PipelineTiming:
    """Comprehensive timing for entire pipeline."""

    data_loading: LoaderTiming
    training_sec: float
    evaluation_sec: float
    inference: InferenceTiming
    output_generation: OutputTiming
    total_sec: float

    def summary(self) -> str:
        """Return human-readable timing summary."""
        lines = [
            "=" * 60,
            "PIPELINE TIMING SUMMARY",
            "=" * 60,
            f"Data Loading:           {self.data_loading.total_sec:>8.2f}s",
            f"  - Parse GeoJSON:      {self.data_loading.parse_geojson_sec:>8.2f}s",
            f"  - Fetch embeddings:   {self.data_loading.fetch_embeddings_sec:>8.2f}s",
            f"  - Stratified split:   {self.data_loading.stratified_split_sec:>8.2f}s",
            f"Training:               {self.training_sec:>8.2f}s",
            f"Evaluation:             {self.evaluation_sec:>8.2f}s",
            f"Inference:              {self.inference.total_sec:>8.2f}s",
            f"  - Batches processed:  {self.inference.batches_processed:>8d}",
            f"  - Embeddings scored:  {self.inference.embeddings_scored:>8d}",
            f"  - Detections found:   {self.inference.detections_found:>8d}",
            f"Output Generation:      {self.output_generation.total_sec:>8.2f}s",
            f"  - Fetch metadata:     {self.output_generation.fetch_metadata_sec:>8.2f}s",
            f"  - Generate tiles:     {self.output_generation.generate_tiles_sec:>8.2f}s",
            f"  - Union tiles:        {self.output_generation.union_tiles_sec:>8.2f}s",
            f"  - Export GeoJSON:     {self.output_generation.export_sec:>8.2f}s",
            "-" * 60,
            f"TOTAL:                  {self.total_sec:>8.2f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Results from classification pipeline run."""

    metrics: EvaluationMetrics
    num_detections: int
    output_files: Dict[str, str]
    timing: PipelineTiming
    train_samples: int
    test_samples: int


class ClassificationPipeline:
    """
    Orchestrates the full classification workflow.

    Usage:
        with ClassificationPipeline(duckdb_path="path/to/db.duckdb") as pipeline:
            result = pipeline.run(
                geojson_path="training.geojson",
                output_dir="output/",
                probability_threshold=0.7,
            )
            print(result.timing.summary())
    """

    def __init__(
        self,
        duckdb_path: str,
        memory_limit: str = "8GB",
    ):
        """
        Initialize pipeline.

        Args:
            duckdb_path: Path to DuckDB database with geo_embeddings table
            memory_limit: DuckDB memory limit (default 8GB for M1 Mac)
        """
        self.duckdb_path = duckdb_path
        self.memory_limit = memory_limit
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _connect(self) -> None:
        """Connect to DuckDB and configure memory settings."""
        self.conn = duckdb.connect(self.duckdb_path, read_only=True)
        self.conn.execute(f"SET memory_limit='{self.memory_limit}'")
        self.conn.execute("SET temp_directory='/tmp'")
        self.conn.execute("LOAD spatial;")

    def _disconnect(self) -> None:
        """Close DuckDB connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "ClassificationPipeline":
        self._connect()
        return self

    def __exit__(self, *args) -> None:
        self._disconnect()

    def run(
        self,
        geojson_path: str,
        output_dir: str,
        test_fraction: float = 0.2,
        probability_threshold: float = 0.5,
        xgb_params: Optional[Dict[str, Any]] = None,
        batch_size: int = 100_000,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """
        Run the full classification pipeline.

        Args:
            geojson_path: Path to training GeoJSON with tile_id, label, class
            output_dir: Directory for output files
            test_fraction: Fraction of data for test set (default 0.2)
            probability_threshold: Threshold for positive classification (default 0.5)
            xgb_params: Optional XGBoost hyperparameters
            batch_size: Batch size for inference (default 100K)
            progress_callback: Optional callback for progress updates

        Returns:
            PipelineResult with metrics, detections, output paths, and timing
        """
        total_start = time.perf_counter()

        # Default XGBoost parameters
        xgb_params = xgb_params or {}
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        default_params.update(xgb_params)

        # =========================================================
        # STEP 1: Load and split training data
        # =========================================================
        self._report_progress(progress_callback, "Loading training data", 0.0)

        loader = ClassificationDataLoader(self.conn, geojson_path)
        train_df, test_df, loader_timing = loader.load(
            test_fraction=test_fraction, random_state=default_params["random_state"]
        )

        # Extract features and labels
        X_train = np.vstack(train_df["embedding"].values).astype(np.float32)
        y_train = train_df["label"].values.astype(np.int32)
        X_test = np.vstack(test_df["embedding"].values).astype(np.float32)
        y_test = test_df["label"].values.astype(np.int32)

        self._report_progress(progress_callback, "Training data loaded", 0.1)
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # =========================================================
        # STEP 2: Train XGBoost classifier
        # =========================================================
        self._report_progress(progress_callback, "Training classifier", 0.15)

        classifier = EmbeddingClassifier(**default_params)
        training_time = classifier.fit(X_train, y_train)

        self._report_progress(progress_callback, "Classifier trained", 0.25)
        print(f"Training time: {training_time:.2f}s")

        # =========================================================
        # STEP 3: Evaluate on test set
        # =========================================================
        self._report_progress(progress_callback, "Evaluating model", 0.3)

        metrics, eval_time = classifier.evaluate(X_test, y_test)

        self._report_progress(progress_callback, "Evaluation complete", 0.35)
        print(f"Test metrics: F1={metrics.f1:.3f}, AUC={metrics.auc_roc:.3f}")

        # =========================================================
        # STEP 4: Run inference on all embeddings
        # =========================================================
        self._report_progress(progress_callback, "Running inference", 0.4)

        inference = BatchInference(
            classifier=classifier,
            duckdb_connection=self.conn,
            batch_size=batch_size,
        )

        total_embeddings = inference.get_total_count()
        print(f"Scoring {total_embeddings:,} embeddings...")

        def inference_progress(scored: int, total: int):
            pct = 0.4 + (scored / total) * 0.4  # 40% to 80%
            self._report_progress(
                progress_callback, f"Inference: {scored:,}/{total:,}", pct
            )

        detections, inference_timing = inference.run(
            probability_threshold=probability_threshold,
            progress_callback=inference_progress,
        )

        self._report_progress(progress_callback, "Inference complete", 0.8)
        print(
            f"Found {len(detections):,} detections above threshold {probability_threshold}"
        )

        # =========================================================
        # STEP 5: Generate output GeoJSON
        # =========================================================
        self._report_progress(progress_callback, "Generating output", 0.85)

        output_generator = OutputGenerator(
            duckdb_connection=self.conn,
            tile_size_px=32,
            tile_overlap_px=16,
            resolution_m=10.0,
        )

        output_paths, output_timing = output_generator.generate_output(
            detections=detections,
            output_dir=output_dir,
            name="classification",
        )

        # Save model
        model_path = os.path.join(output_dir, "model.json")
        classifier.save(model_path)
        output_paths["model"] = model_path

        self._report_progress(progress_callback, "Pipeline complete", 1.0)

        # =========================================================
        # Compile results
        # =========================================================
        total_time = time.perf_counter() - total_start

        timing = PipelineTiming(
            data_loading=loader_timing,
            training_sec=training_time,
            evaluation_sec=eval_time,
            inference=inference_timing,
            output_generation=output_timing,
            total_sec=total_time,
        )

        result = PipelineResult(
            metrics=metrics,
            num_detections=len(detections),
            output_files=output_paths,
            timing=timing,
            train_samples=len(train_df),
            test_samples=len(test_df),
        )

        print(timing.summary())

        return result

    def _report_progress(
        self,
        callback: Optional[Callable[[str, float], None]],
        message: str,
        progress: float,
    ) -> None:
        """Report progress if callback is provided."""
        if callback is not None:
            callback(message, progress)
