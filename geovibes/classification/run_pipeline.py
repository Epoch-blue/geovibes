#!/usr/bin/env python3
"""
Run the classification pipeline from command line.

Usage with single combined GeoJSON:
    uv run python -m geovibes.classification.run_pipeline \
        --geojson full_dataset.geojson \
        --output output/ \
        --db path/to/embeddings.db \
        --threshold 0.5

Usage with separate positives and negatives:
    uv run python -m geovibes.classification.run_pipeline \
        --positives geovibes_labeled.geojson \
        --negatives sampled_negatives.geojson \
        --output output/ \
        --db path/to/embeddings.db \
        --threshold 0.5
"""

import argparse
import json
import tempfile
from pathlib import Path

from geovibes.classification.pipeline import ClassificationPipeline


def combine_datasets(positives_path: str, negatives_path: str) -> str:
    """
    Combine positives and negatives GeoJSONs into a single dataset.

    Parameters
    ----------
    positives_path : str
        Path to GeoJSON with positive examples (label=1)
    negatives_path : str
        Path to GeoJSON with negative examples (label=0)

    Returns
    -------
    str
        Path to combined temporary GeoJSON file
    """
    # Load positives
    with open(positives_path) as f:
        positives = json.load(f)

    # Load negatives
    with open(negatives_path) as f:
        negatives = json.load(f)

    # Ensure all positives have label=1 and class set
    pos_features = []
    for feature in positives.get("features", []):
        props = feature.get("properties", {})
        # Ensure label is 1
        props["label"] = 1
        # Use existing class or default to geovibes_pos
        if "class" not in props:
            props["class"] = "geovibes_pos"
        feature["properties"] = props
        pos_features.append(feature)

    # Ensure all negatives have label=0 and class set
    neg_features = []
    for feature in negatives.get("features", []):
        props = feature.get("properties", {})
        # Ensure label is 0
        props["label"] = 0
        # Use existing class or default to sampled_neg
        if "class" not in props:
            props["class"] = "sampled_neg"
        feature["properties"] = props
        neg_features.append(feature)

    # Combine
    combined = {
        "type": "FeatureCollection",
        "features": pos_features + neg_features,
        "metadata": {
            "positives_source": positives_path,
            "negatives_source": negatives_path,
            "positive_count": len(pos_features),
            "negative_count": len(neg_features),
        },
    }

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False)
    json.dump(combined, temp_file, indent=2)
    temp_file.close()

    print(f"Combined {len(pos_features)} positives + {len(neg_features)} negatives")
    print(f"Temporary combined file: {temp_file.name}")

    return temp_file.name


def main():
    parser = argparse.ArgumentParser(
        description="Run classification pipeline on satellite embeddings"
    )

    # Input options - either single geojson OR positives + negatives
    input_group = parser.add_argument_group("input options")
    input_group.add_argument(
        "--geojson",
        help="Single GeoJSON with both positives and negatives (label column)",
    )
    input_group.add_argument(
        "--positives",
        help="GeoJSON with positive examples (from GeoVibes)",
    )
    input_group.add_argument(
        "--negatives",
        help="GeoJSON with negative examples (from sample_negatives)",
    )

    # Required arguments
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to DuckDB database with geo_embeddings table",
    )

    # Optional arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for detection (default: 0.5)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit (default: 8GB)",
    )

    args = parser.parse_args()

    # Validate input arguments
    if args.geojson and (args.positives or args.negatives):
        parser.error("Use either --geojson OR (--positives and --negatives), not both")

    if not args.geojson and not (args.positives and args.negatives):
        parser.error(
            "Must provide either --geojson OR both --positives and --negatives"
        )

    # Determine input path
    if args.geojson:
        geojson_path = args.geojson
    else:
        # Combine positives and negatives
        geojson_path = combine_datasets(args.positives, args.negatives)

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    with ClassificationPipeline(
        duckdb_path=args.db,
        memory_limit=args.memory_limit,
    ) as pipeline:
        result = pipeline.run(
            geojson_path=geojson_path,
            output_dir=args.output,
            probability_threshold=args.threshold,
            test_fraction=args.test_fraction,
        )

    print(f"\nDetections: {result.num_detections}")
    print(f"F1: {result.metrics.f1:.3f}, AUC: {result.metrics.auc_roc:.3f}")
    print(f"Output files: {result.output_files}")


if __name__ == "__main__":
    main()
