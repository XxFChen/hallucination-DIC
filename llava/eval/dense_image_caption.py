import torch
import os
import json
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images


def eval_single_image(model_path, image_path, question, conv_mode="llava_v1", temperature=0.2, top_p=None, num_beams=1, max_new_tokens=128):
    # 禁用初始化，优化加载速度
    disable_torch_init()

    # 加载模型
    print("Loading model and tokenizer...")
    model_name = os.path.basename(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 处理问题和对话模板
    print("Preparing prompt...")
    if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()

    # 处理图像
    print("Processing image...")
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Tokenize 输入
    input_ids = tokenizer_image_token(
        final_prompt, tokenizer, return_tensors="pt"
    )
    input_ids = input_ids.to(device)
    image_tensor = image_tensor.to(dtype=torch.float16, device=device)

    # 推理
    print("Running inference...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),  # 加入批量维度
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

    # 解码生成的回答
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask about the image.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode for the model.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling probability.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of tokens to generate.")
    args = parser.parse_args()

    # 执行推理
    result = eval_single_image(
        model_path=args.model_path,
        image_path=args.image_path,
        question=args.question,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nGenerated Answer:", result)
