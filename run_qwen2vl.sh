#!/bin/bash
set -e

# Qwen2VL専用評価スクリプト

python eval_screenspot_pro.py  \
    --model_type qwen2vl  \
    --screenspot_imgs "./data/ScreenSpot-Pro/images"  \
    --screenspot_test "./data/ScreenSpot-Pro/annotations"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen2vl.json" \
    --inst_style "instruction" 