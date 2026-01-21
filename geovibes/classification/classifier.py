"""Binary classifiers for embeddings using XGBoost and Linear SVM."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import time
import joblib
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for binary classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    optimal_threshold: float = 0.5


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find threshold that maximizes F1 score using precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities

    Returns
    -------
    float
        Optimal threshold for F1 score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns n+1 precision/recall values, but n thresholds
    precision = precision[:-1]
    recall = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)

    if len(f1_scores) == 0 or f1_scores.max() == 0:
        return 0.5

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


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
        scale_pos_weight: Optional[float] = None,
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
        scale_pos_weight : float, optional
            Ratio of negative to positive samples for class imbalance.
            If None, will be auto-computed from training data.
        """
        self._scale_pos_weight = scale_pos_weight
        self._init_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": tree_method,
        }
        self.model: Optional[XGBClassifier] = None
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

        scale_pos_weight = self._scale_pos_weight
        if scale_pos_weight is None:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos
            else:
                scale_pos_weight = 1.0

        self.model = XGBClassifier(
            **self._init_params,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        fit_time = time.perf_counter() - start_time
        self.fit_time = fit_time
        return fit_time

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[EvaluationMetrics, float]:
        """
        Evaluate the classifier on test data.

        Finds the optimal threshold that maximizes F1 score on the test set,
        then reports metrics at that threshold.

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

        y_proba = self.model.predict_proba(X_test)[:, 1]

        optimal_threshold = find_optimal_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba),
            optimal_threshold=optimal_threshold,
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
        classifier.model = XGBClassifier()
        classifier.model.load_model(path)
        return classifier


class LinearSVMClassifier:
    """
    Linear SVM classifier for embeddings (linear probe).

    Uses LinearSVC with CalibratedClassifierCV for probability estimates.
    This serves as a linear baseline to understand how much structure
    already exists in the embedding space for a given task.
    """

    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        max_iter: int = 10000,
        class_weight: str = "balanced",
        n_jobs: int = -1,
    ):
        """
        Initialize Linear SVM classifier.

        Parameters
        ----------
        C : float
            Regularization parameter (inverse of regularization strength)
        random_state : int
            Random seed for reproducibility
        max_iter : int
            Maximum iterations for solver convergence
        class_weight : str
            'balanced' to handle class imbalance, or None
        n_jobs : int
            Number of parallel threads for calibration (-1 = all cores)
        """
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model: Optional[CalibratedClassifierCV] = None
        self.fit_time: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Train the Linear SVM classifier.

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

        base_svm = LinearSVC(
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            dual="auto",
        )

        self.model = CalibratedClassifierCV(
            estimator=base_svm,
            cv=3,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        fit_time = time.perf_counter() - start_time
        self.fit_time = fit_time
        return fit_time

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[EvaluationMetrics, float]:
        """
        Evaluate the classifier on test data.

        Finds the optimal threshold that maximizes F1 score on the test set,
        then reports metrics at that threshold.

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

        y_proba = self.model.predict_proba(X_test)[:, 1]

        optimal_threshold = find_optimal_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba),
            optimal_threshold=optimal_threshold,
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
        """Save model using joblib."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LinearSVMClassifier":
        """Load model from joblib file."""
        classifier = cls()
        classifier.model = joblib.load(path)
        return classifier

    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """
        Extract the linear weight vector and intercept from the trained SVM.

        The weight vector represents the learned direction in embedding space
        that separates positive from negative samples. For a trained classifier,
        this is essentially the "target concept embedding" - new embeddings
        can be compared to this direction via dot product.

        Since CalibratedClassifierCV trains multiple internal SVMs (one per CV fold),
        this method averages the coefficients across all calibrators.

        Returns
        -------
        Tuple[np.ndarray, float]
            (coef, intercept) where coef has shape (n_features,) and intercept is scalar.
            For prediction: score = X @ coef + intercept (positive = class 1)

        Raises
        ------
        ValueError
            If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        coefs = []
        intercepts = []

        for calibrator in self.model.calibrated_classifiers_:
            base_svm = calibrator.estimator
            coefs.append(base_svm.coef_.flatten())
            intercepts.append(base_svm.intercept_[0])

        avg_coef = np.mean(coefs, axis=0)
        avg_intercept = np.mean(intercepts)

        return avg_coef, avg_intercept

    def get_normalized_direction(self) -> np.ndarray:
        """
        Get the unit-normalized direction vector in embedding space.

        This is the coefficients normalized to unit length, which can be used
        for cosine similarity comparisons with embeddings.

        Returns
        -------
        np.ndarray
            Unit vector of shape (n_features,) pointing toward the positive class
        """
        coef, _ = self.get_coefficients()
        return coef / np.linalg.norm(coef)


ClassifierType = Union[EmbeddingClassifier, LinearSVMClassifier]
