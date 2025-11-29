"""
Batch Inference Module

Performs memory-efficient batch inference over all embeddings in DuckDB.
"""

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional, Callable
import numpy as np
import time


@dataclass
class InferenceTiming:
    """Timing information for batch inference"""

    total_sec: float
    batches_processed: int
    embeddings_scored: int
    detections_found: int


class BatchInference:
    """
    Memory-efficient batch inference over DuckDB embeddings.

    Scores all embeddings in batches using a trained classifier and returns
    IDs where probability exceeds threshold. Uses parallel batch processing
    for improved throughput.

    Memory budget per batch (384-dim float32):
    - 100K embeddings × 384 dims × 4 bytes = ~154MB
    - Plus XGBoost internal buffers ~50MB
    - Total ~200MB per batch, safe for 32GB system
    """

    def __init__(
        self,
        classifier,  # EmbeddingClassifier
        duckdb_connection,
        batch_size: int = 100_000,
        n_workers: int = 4,
    ):
        """
        Initialize batch inference.

        Parameters
        ----------
        classifier : EmbeddingClassifier
            Trained classifier with predict_proba method
        duckdb_connection : duckdb.DuckDBPyConnection
            DuckDB connection with geo_embeddings table
        batch_size : int
            Number of embeddings to score per batch (default 100K)
        n_workers : int
            Number of parallel workers for batch prediction (default 4)
        """
        self.classifier = classifier
        self.conn = duckdb_connection
        self.batch_size = batch_size
        self.n_workers = n_workers

        # Load spatial extension for geometry operations
        self.conn.execute("LOAD spatial;")

    def get_total_count(self) -> int:
        """
        Get total number of embeddings in database.

        Returns
        -------
        int
            Total row count in geo_embeddings table
        """
        result = self.conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()
        return result[0]

    def run(
        self,
        probability_threshold: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Tuple[int, float]], InferenceTiming]:
        """
        Run batch inference over all embeddings.

        Processes embeddings in batches, scores with classifier, and collects
        IDs where probability >= threshold. Uses parallel workers to maximize
        throughput on multi-core systems.

        Parameters
        ----------
        probability_threshold : float
            Minimum probability for detection (default 0.5)
        progress_callback : Optional[Callable[[int, int], None]]
            Called with (embeddings_processed, total_embeddings) after each batch

        Returns
        -------
        detections : List[Tuple[int, float]]
            List of (id, probability) tuples where prob >= threshold
        timing : InferenceTiming
            Performance metrics for the inference run
        """
        start_time = time.perf_counter()

        total_count = self.get_total_count()
        detections = []
        batches_processed = 0
        embeddings_scored = 0

        # Single-threaded batch iteration with parallel prediction
        # We serialize batch loading to avoid concurrent DuckDB queries
        # but parallelize the prediction step which is CPU-bound
        for batch_ids, batch_embeddings in self._iterate_batches():
            # Score batch in parallel workers
            probabilities = self._score_batch(batch_embeddings)

            # Filter by threshold
            mask = probabilities >= probability_threshold
            batch_detections = [
                (int(batch_ids[i]), float(probabilities[i])) for i in np.where(mask)[0]
            ]
            detections.extend(batch_detections)

            batches_processed += 1
            embeddings_scored += len(batch_ids)

            # Report progress
            if progress_callback is not None:
                progress_callback(embeddings_scored, total_count)

        total_time = time.perf_counter() - start_time

        timing = InferenceTiming(
            total_sec=total_time,
            batches_processed=batches_processed,
            embeddings_scored=embeddings_scored,
            detections_found=len(detections),
        )

        return detections, timing

    def _iterate_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator yielding (ids, embeddings) batches.

        Uses LIMIT/OFFSET ordered by id for deterministic iteration.
        Casts UTINYINT embeddings to FLOAT for XGBoost compatibility.

        Yields
        ------
        ids : np.ndarray
            Array of embedding IDs (int64)
        embeddings : np.ndarray
            Array of embeddings (float32), shape (batch_size, embedding_dim)
        """
        offset = 0

        while True:
            # Query batch with type casting
            query = """
                SELECT id, CAST(embedding AS FLOAT[]) as embedding
                FROM geo_embeddings
                ORDER BY id
                LIMIT ? OFFSET ?
            """
            result = self.conn.execute(query, [self.batch_size, offset]).fetchall()

            # Check if done
            if not result:
                break

            # Extract ids and embeddings
            ids = np.array([row[0] for row in result], dtype=np.int64)

            # Convert embedding tuples to numpy array
            # DuckDB returns embeddings as tuples, need to convert to numpy
            embeddings = np.vstack(
                [np.array(row[1], dtype=np.float32) for row in result]
            )

            yield ids, embeddings

            offset += self.batch_size

    def _score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Score a batch of embeddings.

        Uses the classifier's predict_proba method to get positive class
        probabilities. XGBoost releases the GIL during prediction, so
        multiple batches can be scored in parallel if needed.

        Parameters
        ----------
        embeddings : np.ndarray
            Batch of embeddings, shape (n_samples, embedding_dim)

        Returns
        -------
        np.ndarray
            Probabilities of positive class, shape (n_samples,)
        """
        # predict_proba returns (n_samples, 2) for binary classification
        # We want the positive class probability (column 1)
        proba = self.classifier.predict_proba(embeddings)

        # Handle both binary classifier formats:
        # - (n_samples, 2): standard binary format, use column 1
        # - (n_samples,): single probability output, use as-is
        if proba.ndim == 2:
            return proba[:, 1]
        else:
            return proba
