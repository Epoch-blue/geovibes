"""
Batch Inference Module

Performs memory-efficient batch inference over all embeddings in DuckDB.

Optimizations applied:
- Uses fetchdf() instead of fetchall() for 6-7x faster data transfer
- Larger default batch size (500K) for better throughput
- Vectorized numpy operations for threshold filtering
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
    throughput_per_sec: float = 0.0

    def __post_init__(self):
        if self.total_sec > 0:
            self.throughput_per_sec = self.embeddings_scored / self.total_sec


class BatchInference:
    """
    Memory-efficient batch inference over DuckDB embeddings.

    Scores all embeddings in batches using a trained classifier and returns
    IDs where probability exceeds threshold.

    Optimizations:
    - fetchdf() for fast pandas-based data transfer (6-7x faster than fetchall)
    - Larger batch sizes (500K default) for better throughput
    - Vectorized numpy operations

    Memory budget per batch (384-dim float32):
    - 500K embeddings × 384 dims × 4 bytes = ~730MB
    - Plus XGBoost internal buffers ~100MB
    - Total ~830MB per batch, safe for 32GB system
    """

    def __init__(
        self,
        classifier,  # EmbeddingClassifier
        duckdb_connection,
        batch_size: int = 0,  # 0 = auto (single batch if fits in memory)
        max_memory_gb: float = 12.0,  # Max memory for embeddings
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
            Number of embeddings to score per batch.
            0 = auto-detect (use single batch if fits in max_memory_gb)
        max_memory_gb : float
            Maximum memory to use for embeddings (default 12GB, safe for 32GB system)
        """
        self.classifier = classifier
        self.conn = duckdb_connection
        self.max_memory_gb = max_memory_gb
        self._batch_size = batch_size

        # Load spatial extension for geometry operations
        self.conn.execute("LOAD spatial;")

    @property
    def batch_size(self) -> int:
        """Get batch size, auto-detecting if set to 0."""
        if self._batch_size > 0:
            return self._batch_size

        # Auto-detect: use single batch if fits in memory
        total_count = self.get_total_count()
        embedding_dim = self._detect_embedding_dim()

        # Memory estimate: count * dim * 4 bytes (float32)
        memory_gb = total_count * embedding_dim * 4 / (1024**3)

        if memory_gb <= self.max_memory_gb:
            # Single batch fits in memory - fastest option
            return total_count
        else:
            # Use batches that fit in max_memory_gb
            max_batch = int(self.max_memory_gb * (1024**3) / (embedding_dim * 4))
            return max_batch

    def _detect_embedding_dim(self) -> int:
        """Detect embedding dimension from first row."""
        result = self.conn.execute(
            "SELECT embedding FROM geo_embeddings LIMIT 1"
        ).fetchone()
        if result and result[0]:
            return len(result[0])
        return 384  # Default fallback

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

        Processes embeddings in batches using optimized fetchdf() transfer,
        scores with classifier, and collects IDs where probability >= threshold.

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
            # Score batch
            probabilities = self._score_batch(batch_embeddings)

            # Filter by threshold using vectorized operations
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

    def _iterate_batches_fast(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator yielding (ids, embeddings) batches using optimized fetchdf().

        Uses fetchdf() which is 6-7x faster than fetchall() for array data.
        LIMIT/OFFSET pattern ordered by id for deterministic iteration.

        Yields
        ------
        ids : np.ndarray
            Array of embedding IDs (int64)
        embeddings : np.ndarray
            Array of embeddings (float32), shape (batch_size, embedding_dim)
        """
        offset = 0
        query = """
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings
            ORDER BY id
            LIMIT ? OFFSET ?
        """

        while True:
            # Use fetchdf() for fast pandas-based transfer
            df = self.conn.execute(query, [self.batch_size, offset]).fetchdf()

            # Check if done
            if len(df) == 0:
                break

            # Extract ids directly from pandas (already numpy)
            ids = df["id"].values.astype(np.int64)

            # Stack embeddings - pandas stores list columns as object array
            # np.vstack with list comprehension is still needed but faster with fetchdf
            embeddings = np.vstack(df["embedding"].values).astype(np.float32)

            yield ids, embeddings

            offset += self.batch_size

    def _score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Score a batch of embeddings.

        Uses the classifier's predict_proba method to get positive class
        probabilities.

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

        # Handle both binary classifier formats:
        # - (n_samples, 2): standard binary format, use column 1
        # - (n_samples,): single probability output, use as-is
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    # Keep old method for backwards compatibility
    def _iterate_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Legacy method using fetchall() - kept for compatibility."""
        offset = 0
        query = """
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings
            ORDER BY id
            LIMIT ? OFFSET ?
        """

        while True:
            result = self.conn.execute(query, [self.batch_size, offset]).fetchall()

            if not result:
                break

            ids = np.array([row[0] for row in result], dtype=np.int64)
            embeddings = np.vstack(
                [np.array(row[1], dtype=np.float32) for row in result]
            )

            yield ids, embeddings
            offset += self.batch_size
