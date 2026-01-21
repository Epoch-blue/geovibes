"""Batch inference over DuckDB embeddings."""

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional, Callable
import numpy as np
import time


@dataclass
class InferenceTiming:
    """Timing information for batch inference."""

    total_sec: float
    batches_processed: int
    embeddings_scored: int
    detections_found: int
    throughput_per_sec: float = 0.0

    def __post_init__(self):
        if self.total_sec > 0:
            self.throughput_per_sec = self.embeddings_scored / self.total_sec


class BatchInference:
    """
    Batch inference over DuckDB embeddings.

    Scores all embeddings in batches using a trained classifier and returns
    IDs where probability exceeds threshold.
    """

    def __init__(
        self,
        classifier,
        duckdb_connection,
        batch_size: int = 0,
        max_memory_gb: float = 12.0,
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
            Number of embeddings per batch. 0 = auto-detect based on memory.
        max_memory_gb : float
            Maximum memory for embeddings (default 12GB)
        """
        self.classifier = classifier
        self.conn = duckdb_connection
        self.max_memory_gb = max_memory_gb
        self._batch_size = batch_size
        self.conn.execute("LOAD spatial;")

    @property
    def batch_size(self) -> int:
        """Get batch size, auto-detecting if set to 0."""
        if self._batch_size > 0:
            return self._batch_size

        total_count = self.get_total_count()
        embedding_dim = self._detect_embedding_dim()
        memory_gb = total_count * embedding_dim * 4 / (1024**3)

        if memory_gb <= self.max_memory_gb:
            return total_count
        else:
            return int(self.max_memory_gb * (1024**3) / (embedding_dim * 4))

    def _detect_embedding_dim(self) -> int:
        """Detect embedding dimension from first row."""
        result = self.conn.execute(
            "SELECT embedding FROM geo_embeddings LIMIT 1"
        ).fetchone()
        if result and result[0]:
            return len(result[0])
        return 384

    def get_total_count(self) -> int:
        """Get total number of embeddings in database."""
        result = self.conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()
        return result[0]

    def run(
        self,
        probability_threshold: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Tuple[int, float]], InferenceTiming]:
        """
        Run batch inference over all embeddings.

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

        for batch_ids, batch_embeddings in self._iterate_batches_fast():
            probabilities = self._score_batch(batch_embeddings)

            mask = probabilities >= probability_threshold
            if mask.any():
                detected_ids = batch_ids[mask]
                detected_probs = probabilities[mask]
                batch_detections = list(
                    zip(detected_ids.tolist(), detected_probs.tolist())
                )
                detections.extend(batch_detections)

            batches_processed += 1
            embeddings_scored += len(batch_ids)

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

    def _iterate_batches_fast(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator yielding (ids, embeddings) batches using cursor-based pagination.

        Uses WHERE id > last_id instead of OFFSET/LIMIT for O(1) performance
        per batch regardless of position in the table.

        Yields
        ------
        ids : np.ndarray
            Array of embedding IDs (int64)
        embeddings : np.ndarray
            Array of embeddings (float32), shape (batch_size, embedding_dim)
        """
        last_id = -1
        query = """
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings
            WHERE id > ?
            ORDER BY id
            LIMIT ?
        """

        while True:
            df = self.conn.execute(query, [last_id, self.batch_size]).fetchdf()

            if len(df) == 0:
                break

            ids = df["id"].values.astype(np.int64)
            embeddings = np.vstack(df["embedding"].values).astype(np.float32)

            yield ids, embeddings

            last_id = int(ids[-1])

    def _score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Score a batch of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Batch of embeddings, shape (n_samples, embedding_dim)

        Returns
        -------
        np.ndarray
            Probabilities of positive class, shape (n_samples,)
        """
        proba = self.classifier.predict_proba(embeddings)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def _iterate_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Legacy method using fetchall() with cursor-based pagination."""
        last_id = -1
        query = """
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings
            WHERE id > ?
            ORDER BY id
            LIMIT ?
        """

        while True:
            result = self.conn.execute(query, [last_id, self.batch_size]).fetchall()

            if not result:
                break

            ids = np.array([row[0] for row in result], dtype=np.int64)
            embeddings = np.vstack(
                [np.array(row[1], dtype=np.float32) for row in result]
            )

            yield ids, embeddings
            last_id = int(ids[-1])
