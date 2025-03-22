#!/bin/bash

# Set environment variables
export PYTHONPATH=$PWD

# Run the evaluation with Qwen2.5-VL model
python run_qwen25vl.py \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --screenspot_imgs "./data/ScreenSpot-Pro/images" \
    --screenspot_test "./data/ScreenSpot-Pro/annotations" \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen25vl.json" \
    --visualize