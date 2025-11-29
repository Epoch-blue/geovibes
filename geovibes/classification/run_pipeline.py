#!/usr/bin/env python3
"""
Run the classification pipeline from command line.

Usage:
    uv run python -m geovibes.classification.run_pipeline \
        --geojson full_dataset_aquaculture_AL.geojson \
        --output output_aquaculture/ \
        --db local_databases/alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata.db \
        --threshold 0.5
"""

import argparse

from geovibes.classification.pipeline import ClassificationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run classification pipeline on satellite embeddings"
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to training GeoJSON with label and class properties",
    )
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

    with ClassificationPipeline(
        duckdb_path=args.db,
        memory_limit=args.memory_limit,
    ) as pipeline:
        result = pipeline.run(
            geojson_path=args.geojson,
            output_dir=args.output,
            probability_threshold=args.threshold,
            test_fraction=args.test_fraction,
        )

    print(f"\nDetections: {result.num_detections}")
    print(f"F1: {result.metrics.f1:.3f}, AUC: {result.metrics.auc_roc:.3f}")
    print(f"Output files: {result.output_files}")


if __name__ == "__main__":
    main()
