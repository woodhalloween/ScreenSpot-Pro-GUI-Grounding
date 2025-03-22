#!/bin/bash
set -e

# Qwen2.5VL専用評価スクリプト

python eval_screenspot_pro.py  \
    --model_type qwen25vl  \
    --screenspot_imgs "./data/ScreenSpot-Pro/images"  \
    --screenspot_test "./data/ScreenSpot-Pro/annotations"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen25vl.json" \
    --inst_style "instruction"