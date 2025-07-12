#!/bin/bash

# Example script to run embedding generation on Modal using config file

# Example 1: Run with config file (recommended approach)
modal run src/geotiff/modal_embedding.py \
    --config config/modal_embedding_example.json

# Example 2: Run with config file and override some parameters
modal run src/geotiff/modal_embedding.py \
    --config config/modal_embedding_example.json \
    --batch-size 32 \
    --model-name "efficientnet_b0"

# Example 3: Run with config file and override paths
modal run src/geotiff/modal_embedding.py \
    --config config/modal_embedding_example.json \
    --tiles-dir "tiles_new" \
    --output-base-path "embeddings_v2"