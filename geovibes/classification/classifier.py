"""Binary classifier for embeddings using XGBoost."""

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
class EvaluationMetrics:
    """Evaluation metrics for binary classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float


class EmbeddingClassifier:
    """XGBoost binary classifier for embeddings."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
    ):
        """
        Initialize classifier.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Boosting learning rate
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel threads (-1 = all cores)
        tree_method : str
            Tree construction algorithm
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
        """
        Train the classifier.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features)
        y_train : np.ndarray
            Training labels, shape (n_samples,)
        sample_weight : np.ndarray, optional
            Sample weights, shape (n_samples,)

        Returns
        -------
        float
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
        """
        Evaluate the classifier on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test features, shape (n_samples, n_features)
        y_test : np.ndarray
            Test labels, shape (n_samples,)

        Returns
        -------
        Tuple[EvaluationMetrics, float]
            Metrics and evaluation time in seconds
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
        """
        Predict probability of positive class.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Probabilities of positive class, shape (n_samples,)
        """
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Save model to JSON format."""
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "EmbeddingClassifier":
        """Load model from JSON file."""
        classifier = cls()
        classifier.model.load_model(path)
        return classifier
