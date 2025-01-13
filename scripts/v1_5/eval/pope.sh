#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /root/autodl-tmp/llava-v1.6-vicuna-7b \
    --question-file /root/autodl-tmp/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /root/autodl-tmp/LLaVA/playground/data/eval/pope/val2014\
    --answers-file ./playground/data/eval/pope/answers/llava-v1.6-vicuna-7b_layers_0-5.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
     --annotation-dir /root/autodl-tmp/LLaVA/playground/data/eval/pope/coco \
    --question-file /root/autodl-tmp/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /root/autodl-tmp/LLaVA/playground/data/eval/pope/answers/llava-v1.6-vicuna-7b_layers_0-5.jsonl