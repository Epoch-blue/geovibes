#!/bin/bash

# Example script to run embedding generation on Modal using config file

# Example 1: Run with ROI-based config file (recommended approach)
modal run src/geotiff/modal_inference.py \
    --config config/modal_embedding_example.json

# Example 2: Run with ROI file override
modal run src/geotiff/modal_inference.py \
    --config config/modal_embedding_example.json \
    --roi-file geometries/alabama.geojson

# Example 3: Run with config file and override parameters
modal run src/geotiff/modal_inference.py \
    --config config/modal_embedding_example.json \
    --batch-size 32 \
    --model-name "efficientnet_b0"

# Example 4: Run with config file and override paths
modal run src/geotiff/modal_inference.py \
    --config config/modal_embedding_example.json \
    --tiles-dir "tiles_new" \
    --output-base-path "embeddings_v2"