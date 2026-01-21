"""Cross-validation with spatial fold assignment for geospatial classification."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

from geovibes.classification.classifier import (
    EmbeddingClassifier,
    LinearSVMClassifier,
    EvaluationMetrics,
    ClassifierType,
)


@dataclass
class CVResult:
    """Cross-validation results with mean and std for each metric."""

    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_roc_mean: float
    auc_roc_std: float
    fold_metrics: List[EvaluationMetrics]
    fold_train_counts: List[Tuple[int, int]]  # (pos, neg) per fold
    fold_test_counts: List[Tuple[int, int]]
    n_clusters: int
    cluster_distribution: Dict[int, int]  # fold -> n_clusters

    def summary(self) -> str:
        """Return formatted summary of CV results."""
        lines = [
            "=" * 65,
            "SPATIAL 5-FOLD CROSS-VALIDATION RESULTS",
            "=" * 65,
            "Metric       Mean ± Std",
            "-" * 65,
            f"Accuracy:    {self.accuracy_mean:.4f} ± {self.accuracy_std:.4f}",
            f"Precision:   {self.precision_mean:.4f} ± {self.precision_std:.4f}",
            f"Recall:      {self.recall_mean:.4f} ± {self.recall_std:.4f}",
            f"F1:          {self.f1_mean:.4f} ± {self.f1_std:.4f}",
            f"AUC-ROC:     {self.auc_roc_mean:.4f} ± {self.auc_roc_std:.4f}",
            "-" * 65,
            f"Spatial clusters: {self.n_clusters} (distributed across {len(self.cluster_distribution)} folds)",
            "-" * 65,
            "Per-fold breakdown:",
        ]
        for i, m in enumerate(self.fold_metrics):
            train_pos, train_neg = self.fold_train_counts[i]
            test_pos, test_neg = self.fold_test_counts[i]
            n_clusters = self.cluster_distribution.get(i, 0)
            lines.append(
                f"  Fold {i + 1}: F1={m.f1:.4f}, AUC={m.auc_roc:.4f} | "
                f"train={train_pos}+/{train_neg}- | test={test_pos}+/{test_neg}- | "
                f"clusters={n_clusters}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)


def create_spatial_folds(
    df: pd.DataFrame,
    geometries: gpd.GeoSeries,
    n_folds: int = 5,
    buffer_m: float = 500.0,
) -> Tuple[np.ndarray, int, Dict[int, int]]:
    """
    Create fold assignments based on spatial clustering of positives.

    Algorithm:
    1. Buffer positive point geometries and union into contiguous clusters
    2. Explode union into separate polygon clusters
    3. Assign each positive to its containing cluster
    4. Distribute clusters across folds (greedy balancing by sample count)
    5. Assign negatives to fold of spatially nearest positive cluster

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'label' column (1=positive, 0=negative)
    geometries : gpd.GeoSeries
        Point geometries for each sample (aligned with df index)
    n_folds : int
        Number of folds (default 5)
    buffer_m : float
        Buffer distance in meters for clustering nearby positives

    Returns
    -------
    fold_assignments : np.ndarray
        Fold index (0 to n_folds-1) for each sample
    n_clusters : int
        Number of spatial clusters identified
    cluster_distribution : Dict[int, int]
        Number of clusters assigned to each fold
    """
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometries.values, crs="EPSG:4326")
    gdf["_orig_idx"] = np.arange(len(gdf))

    positives = gdf[gdf["label"] == 1].copy()
    negatives = gdf[gdf["label"] == 0].copy()

    if len(positives) == 0:
        raise ValueError("No positive samples for spatial grouping")

    # Determine UTM zone from centroid of positives
    center = positives.geometry.union_all().centroid
    utm_zone = int(((center.x + 180) / 6) + 1)
    hemisphere = "N" if center.y >= 0 else "S"
    utm_epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

    print(f"Projecting to UTM zone {utm_zone}{hemisphere} (EPSG:{utm_epsg})")

    # Project to UTM for accurate buffering
    positives_utm = positives.to_crs(f"EPSG:{utm_epsg}")

    # Buffer and union positive points into clusters
    buffered = positives_utm.geometry.buffer(buffer_m)
    union_geom = unary_union(buffered)

    # Explode into separate polygon clusters
    if union_geom.geom_type == "Polygon":
        polygons = [union_geom]
    elif union_geom.geom_type == "MultiPolygon":
        polygons = list(union_geom.geoms)
    else:
        polygons = [union_geom]

    n_clusters = len(polygons)
    print(f"Created {n_clusters} spatial clusters from {len(positives)} positives")

    # Create GeoDataFrame of polygon clusters
    clusters_gdf = gpd.GeoDataFrame(
        {"cluster_id": range(n_clusters)}, geometry=polygons, crs=f"EPSG:{utm_epsg}"
    )

    # Assign each positive to a cluster via spatial join
    positives_utm = positives_utm.copy()
    joined = gpd.sjoin(positives_utm, clusters_gdf, how="left", predicate="within")
    positives_utm["cluster_id"] = joined["cluster_id"].fillna(0).astype(int)

    # Handle any positives not within a cluster (edge case - assign to nearest)
    unassigned = positives_utm[positives_utm["cluster_id"].isna()]
    if len(unassigned) > 0:
        for idx in unassigned.index:
            point = positives_utm.loc[idx, "geometry"]
            distances = clusters_gdf.geometry.distance(point)
            nearest = distances.idxmin()
            positives_utm.at[idx, "cluster_id"] = clusters_gdf.loc[
                nearest, "cluster_id"
            ]

    # Distribute clusters across folds, balancing sample counts
    cluster_counts = (
        positives_utm.groupby("cluster_id").size().sort_values(ascending=False)
    )
    cluster_to_fold = {}
    fold_counts = [0] * n_folds
    cluster_distribution = {i: 0 for i in range(n_folds)}

    for cluster_id, count in cluster_counts.items():
        # Assign to fold with fewest samples (greedy balancing)
        min_fold = int(np.argmin(fold_counts))
        cluster_to_fold[cluster_id] = min_fold
        fold_counts[min_fold] += count
        cluster_distribution[min_fold] += 1

    print(f"Fold sample distribution: {fold_counts}")

    # Assign fold to each positive
    positives_utm["fold"] = positives_utm["cluster_id"].map(cluster_to_fold)

    # For negatives: assign to fold of spatially nearest positive cluster
    fold_assignments = np.zeros(len(gdf), dtype=int)

    # Set positive fold assignments
    for idx, row in positives_utm.iterrows():
        orig_idx = gdf.loc[idx, "_orig_idx"]
        fold_assignments[orig_idx] = row["fold"]

    if len(negatives) > 0:
        negatives_utm = negatives.to_crs(f"EPSG:{utm_epsg}")

        # For each negative, find nearest cluster and assign same fold
        for idx, row in negatives_utm.iterrows():
            point = row.geometry
            distances = clusters_gdf.geometry.distance(point)
            nearest_cluster = clusters_gdf.loc[distances.idxmin(), "cluster_id"]
            fold = cluster_to_fold.get(nearest_cluster, 0)

            orig_idx = gdf.loc[idx, "_orig_idx"]
            fold_assignments[orig_idx] = fold

    return fold_assignments, n_clusters, cluster_distribution


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    fold_assignments: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_folds: int = 5,
    n_clusters: int = 0,
    cluster_distribution: Optional[Dict[int, int]] = None,
    classifier_type: str = "xgboost",
    **xgb_params,
) -> CVResult:
    """
    Perform k-fold cross-validation with pre-assigned folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    fold_assignments : np.ndarray
        Fold index for each sample (0 to n_folds-1)
    sample_weights : np.ndarray, optional
        Sample weights for training
    n_folds : int
        Number of folds
    n_clusters : int
        Number of spatial clusters (for reporting)
    cluster_distribution : Dict[int, int], optional
        Clusters per fold (for reporting)
    classifier_type : str
        Type of classifier: 'xgboost' or 'linear-svm'
    **xgb_params
        XGBoost hyperparameters (only used if classifier_type='xgboost')

    Returns
    -------
    CVResult
        Cross-validation results with mean/std metrics
    """
    fold_metrics = []
    fold_train_counts = []
    fold_test_counts = []

    for k in range(n_folds):
        test_mask = fold_assignments == k
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Skip fold if test set is empty or has only one class
        if len(X_test) == 0:
            print(f"  Fold {k + 1}: Skipping (empty test set)")
            continue

        unique_test_labels = np.unique(y_test)
        if len(unique_test_labels) < 2:
            print(
                f"  Fold {k + 1}: Skipping (test set has only class {unique_test_labels[0]})"
            )
            continue

        weights = None
        if sample_weights is not None:
            weights = sample_weights[train_mask]

        classifier: ClassifierType
        if classifier_type == "linear-svm":
            random_state = xgb_params.get("random_state", 42)
            classifier = LinearSVMClassifier(random_state=random_state)
        else:
            classifier = EmbeddingClassifier(**xgb_params)

        classifier.fit(X_train, y_train, sample_weight=weights)
        metrics, _ = classifier.evaluate(X_test, y_test)

        train_pos = int((y_train == 1).sum())
        train_neg = int((y_train == 0).sum())
        test_pos = int((y_test == 1).sum())
        test_neg = int((y_test == 0).sum())

        print(
            f"  Fold {k + 1}: F1={metrics.f1:.4f}, AUC={metrics.auc_roc:.4f} "
            f"(train: {train_pos}+/{train_neg}-, test: {test_pos}+/{test_neg}-)"
        )

        fold_metrics.append(metrics)
        fold_train_counts.append((train_pos, train_neg))
        fold_test_counts.append((test_pos, test_neg))

    if len(fold_metrics) == 0:
        raise ValueError("No valid folds - check data distribution")

    # Compute mean and std for each metric
    accuracies = [m.accuracy for m in fold_metrics]
    precisions = [m.precision for m in fold_metrics]
    recalls = [m.recall for m in fold_metrics]
    f1s = [m.f1 for m in fold_metrics]
    aucs = [m.auc_roc for m in fold_metrics]

    return CVResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        precision_mean=float(np.mean(precisions)),
        precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
        auc_roc_mean=float(np.mean(aucs)),
        auc_roc_std=float(np.std(aucs)),
        fold_metrics=fold_metrics,
        fold_train_counts=fold_train_counts,
        fold_test_counts=fold_test_counts,
        n_clusters=n_clusters,
        cluster_distribution=cluster_distribution or {},
    )
