python -m llava.eval.model_vqa \
    --model-path /root/autodl-tmp/llava-v1.6-vicuna-7b \
    --question-file /root/autodl-tmp/dense_image/question/question.jsonl \
    --image-folder /root/autodl-tmp/dense_image/images \
    --answers-file /root/autodl-tmp/dense_image/answer/llava-v1.6-vicuna-7b_answer-improve.json \
    --temperature 0 \
    --conv-mode vicuna_v1
