#!/usr/bin/env python3
"""
Validate model detections against truth points.

Computes spatial accuracy metrics by checking how many detections
fall within a buffer distance of truth points.

Usage:
    uv run scripts/validate_detections.py \
        --detections classification_output_v5/classification_detections.geojson \
        --truth truth_dataset_palm_oil.geojson \
        --buffer 500

    # With probability threshold filter
    uv run scripts/validate_detections.py \
        --detections classification_output_v5/classification_detections.geojson \
        --truth truth_dataset_palm_oil.geojson \
        --buffer 500 \
        --min-prob 0.7
"""

import argparse

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union


def load_detections(path: str, min_probability: float = 0.0) -> gpd.GeoDataFrame:
    """Load detections GeoJSON, optionally filtering by probability."""
    gdf = gpd.read_file(path)
    if "probability" in gdf.columns and min_probability > 0:
        original_count = len(gdf)
        gdf = gdf[gdf["probability"] >= min_probability].copy()
        print(f"Filtered detections: {original_count} -> {len(gdf)} (prob >= {min_probability})")
    return gdf


def load_truth(path: str) -> gpd.GeoDataFrame:
    """Load truth points GeoJSON, filtering to positives only."""
    gdf = gpd.read_file(path)
    if "label" in gdf.columns:
        gdf = gdf[gdf["label"] == 1].copy()
    print(f"Truth points (positives): {len(gdf)}")
    return gdf


def get_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Determine appropriate UTM CRS from geometry centroid."""
    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    utm_zone = int(((center_lon + 180) / 6) + 1)
    hemisphere = "N" if center_lat >= 0 else "S"
    epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
    
    print(f"Using UTM zone {utm_zone}{hemisphere} (EPSG:{epsg})")
    return f"EPSG:{epsg}"


def compute_detection_distances(
    detections: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    utm_crs: str,
) -> np.ndarray:
    """Compute distance from each detection centroid to nearest truth point."""
    det_utm = detections.to_crs(utm_crs).copy()
    truth_utm = truth.to_crs(utm_crs).copy()
    
    det_centroids = det_utm.geometry.centroid
    
    distances = []
    for centroid in det_centroids:
        min_dist = truth_utm.geometry.distance(centroid).min()
        distances.append(min_dist)
    
    return np.array(distances)


def compute_truth_coverage(
    detections: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    buffer_m: float,
    utm_crs: str,
) -> tuple[int, int, list[int]]:
    """
    Compute how many truth points have at least one detection within buffer.
    
    Returns:
        covered_count: Number of truth points with nearby detections
        total_truth: Total truth points
        covered_indices: Indices of covered truth points
    """
    det_utm = detections.to_crs(utm_crs).copy()
    truth_utm = truth.to_crs(utm_crs).copy()
    
    det_centroids = det_utm.geometry.centroid
    det_buffered = unary_union(det_centroids.buffer(buffer_m))
    
    covered_indices = []
    for idx, row in truth_utm.iterrows():
        if row.geometry.within(det_buffered) or row.geometry.intersects(det_buffered):
            covered_indices.append(idx)
    
    return len(covered_indices), len(truth_utm), covered_indices


def validate(
    detections_path: str,
    truth_path: str,
    buffer_m: float = 500.0,
    min_probability: float = 0.0,
) -> dict:
    """
    Validate detections against truth points.
    
    Returns dict with:
        - true_positives: Detections within buffer of a truth point
        - false_positives: Detections not near any truth point
        - truth_covered: Truth points with at least one nearby detection
        - truth_missed: Truth points with no nearby detections
        - precision: TP / (TP + FP)
        - recall: truth_covered / total_truth
        - f1: Harmonic mean of precision and recall
        - distance_stats: Statistics on detection-to-truth distances
    """
    print(f"\n{'='*60}")
    print("DETECTION VALIDATION")
    print(f"{'='*60}")
    print(f"Detections: {detections_path}")
    print(f"Truth:      {truth_path}")
    print(f"Buffer:     {buffer_m}m")
    if min_probability > 0:
        print(f"Min prob:   {min_probability}")
    print(f"{'='*60}\n")
    
    detections = load_detections(detections_path, min_probability)
    truth = load_truth(truth_path)
    
    if len(detections) == 0:
        print("No detections to validate!")
        return {}
    
    if len(truth) == 0:
        print("No truth points to validate against!")
        return {}
    
    utm_crs = get_utm_crs(truth)
    
    print("\nComputing distances...")
    distances = compute_detection_distances(detections, truth, utm_crs)
    
    true_positives = int((distances <= buffer_m).sum())
    false_positives = int((distances > buffer_m).sum())
    
    print("Computing truth coverage...")
    truth_covered, total_truth, _ = compute_truth_coverage(
        detections, truth, buffer_m, utm_crs
    )
    truth_missed = total_truth - truth_covered
    
    precision = true_positives / len(detections) if len(detections) > 0 else 0
    recall = truth_covered / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        "total_detections": len(detections),
        "total_truth": total_truth,
        "buffer_m": buffer_m,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "truth_covered": truth_covered,
        "truth_missed": truth_missed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "distance_stats": {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "std": float(distances.std()),
            "p25": float(np.percentile(distances, 25)),
            "p75": float(np.percentile(distances, 75)),
            "p90": float(np.percentile(distances, 90)),
            "p95": float(np.percentile(distances, 95)),
        },
    }
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print("\nDetection counts:")
    print(f"  Total detections:    {results['total_detections']:,}")
    print(f"  True positives:      {results['true_positives']:,} (within {buffer_m}m of truth)")
    print(f"  False positives:     {results['false_positives']:,} (not near any truth)")
    
    print("\nTruth coverage:")
    print(f"  Total truth points:  {results['total_truth']:,}")
    print(f"  Covered by detect:   {results['truth_covered']:,}")
    print(f"  Missed:              {results['truth_missed']:,}")
    
    print("\nMetrics:")
    print(f"  Precision:           {results['precision']:.3f} ({true_positives}/{len(detections)})")
    print(f"  Recall:              {results['recall']:.3f} ({truth_covered}/{total_truth})")
    print(f"  F1 Score:            {results['f1']:.3f}")
    
    print("\nDistance statistics (detection to nearest truth):")
    ds = results['distance_stats']
    print(f"  Min:                 {ds['min']:,.0f}m")
    print(f"  25th percentile:     {ds['p25']:,.0f}m")
    print(f"  Median:              {ds['median']:,.0f}m")
    print(f"  Mean:                {ds['mean']:,.0f}m")
    print(f"  75th percentile:     {ds['p75']:,.0f}m")
    print(f"  90th percentile:     {ds['p90']:,.0f}m")
    print(f"  95th percentile:     {ds['p95']:,.0f}m")
    print(f"  Max:                 {ds['max']:,.0f}m")
    print(f"  Std dev:             {ds['std']:,.0f}m")
    
    within_thresholds = [100, 250, 500, 1000, 2000, 5000]
    print("\nDetections within distance thresholds:")
    for thresh in within_thresholds:
        count = int((distances <= thresh).sum())
        pct = count / len(distances) * 100
        print(f"  ≤{thresh:,}m: {count:,} ({pct:.1f}%)")
    
    print(f"\n{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate model detections against truth points"
    )
    parser.add_argument(
        "--detections", "-d",
        required=True,
        help="Path to detections GeoJSON (output from classification pipeline)",
    )
    parser.add_argument(
        "--truth", "-t",
        required=True,
        help="Path to truth points GeoJSON (with label=1 for positives)",
    )
    parser.add_argument(
        "--buffer", "-b",
        type=float,
        default=500.0,
        help="Buffer distance in meters for matching (default: 500)",
    )
    parser.add_argument(
        "--min-prob", "-p",
        type=float,
        default=0.0,
        help="Minimum probability threshold for detections (default: 0, use all)",
    )
    
    args = parser.parse_args()
    
    validate(
        detections_path=args.detections,
        truth_path=args.truth,
        buffer_m=args.buffer,
        min_probability=args.min_prob,
    )


if __name__ == "__main__":
    main()
