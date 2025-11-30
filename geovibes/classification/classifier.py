"""Binary classifier for embeddings using XGBoost.

Optimized for M1 Mac with histogram-based tree method and parallel processing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassifierTiming:
    """Timing information for training and evaluation."""

    fit_sec: float
    evaluate_sec: float


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for binary classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float


class EmbeddingClassifier:
    """XGBoost binary classifier wrapper for embeddings.

    Optimized for M1 Mac using histogram-based tree method and parallel processing.
    No try-except blocks - fails fast on errors.

    Attributes:
        model: XGBClassifier instance
        fit_time: Training time in seconds (None until fit() is called)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
    ):
        """Initialize classifier with XGBoost parameters.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel threads (-1 = use all cores)
            tree_method: Tree construction algorithm ("hist" for M1 optimization)
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method=tree_method,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        self.fit_time: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Train the classifier.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            sample_weight: Optional sample weights, shape (n_samples,)

        Returns:
            Training time in seconds
        """
        start_time = time.perf_counter()
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        fit_time = time.perf_counter() - start_time
        self.fit_time = fit_time
        return fit_time

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[EvaluationMetrics, float]:
        """Evaluate the classifier on test data.

        Args:
            X_test: Test features, shape (n_samples, n_features)
            y_test: Test labels, shape (n_samples,)

        Returns:
            Tuple of (EvaluationMetrics, evaluation_time_sec)
        """
        start_time = time.perf_counter()

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            auc_roc=roc_auc_score(y_test, y_proba),
        )

        eval_time = time.perf_counter() - start_time

        return metrics, eval_time

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Probabilities of positive class, shape (n_samples,)
        """
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Save model to JSON format.

        Args:
            path: File path to save model (should end with .json)
        """
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "EmbeddingClassifier":
        """Load model from JSON file.

        Args:
            path: File path to load model from

        Returns:
            EmbeddingClassifier instance with loaded model
        """
        classifier = cls()
        classifier.model.load_model(path)
        return classifier
