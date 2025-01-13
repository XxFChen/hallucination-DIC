import json
import os
import random

def create_question_file(coco_path, output_file, num_samples=200, random_seed=42):
    # 加载COCO验证集信息
    val_data = json.load(open(os.path.join(coco_path, 'instances_val2014.json')))
    
    # 从所有图片中随机采样
    sampled_images = random.sample(val_data['images'], num_samples)
    
    questions = []
    for i, img in enumerate(sampled_images):
        questions.append({
            'image': f"COCO_val2014_{img['id']:012d}.jpg",
            'id': img['id'],
            'question_id': i,  # 添加question_id
            'text': "Please describe this image in detail."  # 添加固定的问题文本
        })
    
    # 保存为jsonl格式
    with open(output_file, 'w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-path', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    
    create_question_file(args.coco_path, args.output_file, 
                        args.num_samples, args.random_seed)