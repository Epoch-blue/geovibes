#!/usr/bin/env python3
"""
Standalone classification pipeline test script.

Imports classification modules directly without triggering geovibes UI imports.
"""

import importlib.util
import os
import sys
import time
import warnings

import duckdb
import numpy as np

warnings.filterwarnings("ignore")

# Add the classification module path directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DIR = os.path.join(SCRIPT_DIR, "geovibes", "classification")
sys.path.insert(0, CLASSIFICATION_DIR)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load modules
data_loader = load_module(
    "data_loader", os.path.join(CLASSIFICATION_DIR, "data_loader.py")
)
classifier = load_module(
    "classifier", os.path.join(CLASSIFICATION_DIR, "classifier.py")
)
inference = load_module("inference", os.path.join(CLASSIFICATION_DIR, "inference.py"))
output = load_module("output", os.path.join(CLASSIFICATION_DIR, "output.py"))

print("=" * 60)
print("CLASSIFICATION PIPELINE TEST")
print("=" * 60)

total_start = time.perf_counter()

# Configuration
DUCKDB_PATH = "local_databases/alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata.db"
GEOJSON_PATH = "test_data/sample_training_data.geojson"
OUTPUT_DIR = "test_data/classification_output"

# Step 1: Connect
print("\n[1] Connecting to database...")
conn = duckdb.connect(DUCKDB_PATH, read_only=True)
conn.execute("SET memory_limit='8GB'")
conn.execute("LOAD spatial;")

# Step 2: Load data
print("\n[2] Loading training data...")
loader = data_loader.ClassificationDataLoader(conn, GEOJSON_PATH)
train_df, test_df, loader_timing = loader.load(test_fraction=0.2)
print(f"    Train: {len(train_df)}, Test: {len(test_df)}")
print(f"    Time: {loader_timing.total_sec:.2f}s")

# Step 3: Prepare features
print("\n[3] Preparing features...")
X_train = np.vstack(train_df["embedding"].values).astype(np.float32)
y_train = train_df["label"].values.astype(np.int32)
X_test = np.vstack(test_df["embedding"].values).astype(np.float32)
y_test = test_df["label"].values.astype(np.int32)
print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

# Step 4: Train
print("\n[4] Training classifier...")
clf = classifier.EmbeddingClassifier(n_estimators=50, max_depth=4)
train_time = clf.fit(X_train, y_train)
print(f"    Training time: {train_time:.2f}s")

# Step 5: Evaluate
print("\n[5] Evaluating on test set...")
metrics, eval_time = clf.evaluate(X_test, y_test)
print(f"    Accuracy:  {metrics.accuracy:.3f}")
print(f"    Precision: {metrics.precision:.3f}")
print(f"    Recall:    {metrics.recall:.3f}")
print(f"    F1:        {metrics.f1:.3f}")
print(f"    AUC-ROC:   {metrics.auc_roc:.3f}")
print(f"    Time: {eval_time:.2f}s")

# Step 6: Full inference
print("\n[6] Running inference on ALL embeddings...")
batch_inference = inference.BatchInference(
    classifier=clf, duckdb_connection=conn, batch_size=100_000
)
total_count = batch_inference.get_total_count()
print(f"    Total embeddings: {total_count:,}")


def progress_callback(processed, total):
    pct = processed / total * 100
    print(f"\r    Progress: {processed:,}/{total:,} ({pct:.1f}%)", end="", flush=True)


detections, inference_timing = batch_inference.run(
    probability_threshold=0.5, progress_callback=progress_callback
)
print()  # newline after progress
print(f"    Detections found: {inference_timing.detections_found:,}")
print(f"    Batches processed: {inference_timing.batches_processed}")
print(f"    Time: {inference_timing.total_sec:.2f}s")
print(f"    Throughput: {total_count/inference_timing.total_sec:,.0f} embeddings/sec")

# Step 7: Generate output (if detections exist)
if len(detections) > 0:
    print("\n[7] Generating output GeoJSON...")
    output_gen = output.OutputGenerator(duckdb_connection=conn)
    output_paths, output_timing = output_gen.generate_output(
        detections=detections, output_dir=OUTPUT_DIR, name="classification"
    )
    print(f"    Fetch metadata: {output_timing.fetch_metadata_sec:.2f}s")
    print(f"    Generate tiles: {output_timing.generate_tiles_sec:.2f}s")
    print(f"    Union tiles:    {output_timing.union_tiles_sec:.2f}s")
    print(f"    Export:         {output_timing.export_sec:.2f}s")
    print(f"    Total:          {output_timing.total_sec:.2f}s")
    print("\n    Output files:")
    for name, path in output_paths.items():
        print(f"      {name}: {path}")
else:
    print("\n[7] No detections - skipping output generation")
    output_timing = None

# Save model
print("\n[8] Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, "model.json")
clf.save(model_path)
print(f"    Saved to: {model_path}")

conn.close()

total_time = time.perf_counter() - total_start

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"Data loading:    {loader_timing.total_sec:>8.2f}s")
print(f"Training:        {train_time:>8.2f}s")
print(f"Evaluation:      {eval_time:>8.2f}s")
print(f"Inference:       {inference_timing.total_sec:>8.2f}s")
if output_timing:
    print(f"Output gen:      {output_timing.total_sec:>8.2f}s")
print("-" * 30)
print(f"TOTAL:           {total_time:>8.2f}s")
print("=" * 60)
print("\nSUCCESS!")
