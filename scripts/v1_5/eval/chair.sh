#!/bin/bash

# 设置路径
COCO_PATH="/root/autodl-tmp/annotations"  # COCO注释文件目录
COCO_IMAGES="/root/autodl-tmp/LLaVA/playground/data/eval/pope/val2014"  # COCO验证集图片目录
MODEL_PATH="/root/autodl-tmp/llava-v1.6-vicuna-7b"
OUTPUT_DIR="playground/data/eval/chair"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 1. 准备问题文件
# python /root/autodl-tmp/LLaVA/llava/eval/prepare_chair_questions.py \
#     --coco-path $COCO_PATH \
#     --output-file $OUTPUT_DIR/chair_questions.jsonl \
#     --num-samples 200 \
#     --random-seed 42

# 2. 运行模型生成描述
python  /root/autodl-tmp/LLaVA/llava/eval/model_chair_loader.py \
    --model-path "$MODEL_PATH" \
    --question-file "$OUTPUT_DIR/chair_questions.jsonl" \
    --image-folder "$COCO_IMAGES" \
    --answers-file "$OUTPUT_DIR/predictions.json" \
    --temperature 0 \
    --conv-mode vicuna_v1

# 3. 运行CHAIR评测
python llava/eval/chair.py \
    --cap_file $OUTPUT_DIR/predictions.jsonl \
    --image_id_key "image_id" \
    --caption_key "text" \
    --coco_path $COCO_PATH \
    --save_path $OUTPUT_DIR/chair_results.jsonl