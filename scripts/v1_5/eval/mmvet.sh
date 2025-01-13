#!/bin/bash
# mkdir -p ./playground/data/eval/mm-vet/answers
# mkdir -p ./playground/data/eval/mm-vet/results

python -m llava.eval.model_vqa \
    --model-path /root/autodl-tmp/llava-v1.6-vicuna-7b \
    --question-file /root/autodl-tmp/MMvet/mm-vet/mm-vet.jsonl \
    --image-folder /root/autodl-tmp/MMvet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.6-vicuna-7b_test_1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.6-vicuna-7b_test_1.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.6-vicuna-7b_test_1.json

python tran.py





