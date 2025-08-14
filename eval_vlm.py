import argparse
import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig

warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model(
        lm_config, 
        device, 
):
    tokenizer = AutoTokenizer.from_pretrained('./model')
    if args.load == 0: # 当load==0是使用pytorch导入模型
        moe_path = '_moe' if args.use_moe else ""
        modes = {0: "pretrain_vlm", 1: "sft_vlm", 2: "sft_vlm_multi"}
        ckp = f"./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth"
        model = MiniMindVLM(
            lm_config, 
            vision_model_path="./model/vision_model/clip-vit-base-patch16"
        )
        state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict(
            {k:v for k,v in state_dict.item() if "mask" not in k}, 
            strict=False
        )

    else:
        # 用transformers库加载模型
        transformers_model_path = "MiniMind2-Small-V"
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model("./model/vision_model/clip-vit-base-patch16")

    print(f"VLM参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")

    vision_model, preprocess = model.vision_encoder, model.processor
    return model.eval().to(device), tokenizer, vision_model.eval().to(device), preprocess

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_name", default="None", type=str)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--temperature", default=0.65, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    # small model: hidden_size=512, num_hidden_layers=8
    # mid model: hidden_size=768, num_hidden_layers=16
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=8192, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)

    parser.add_argument("--use_multi", default=1, type=int) # 1为单图推理，2为多图推理
    parser.add_argument("--stream", default=True, type=bool) # 是否进行流式输出。流式输出会动态的输出模型生成的结果。非流式输出会在模型生成完整的内容后一次性的输出。
    parser.add_argument("--load", default=1, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument("--model_mode", default=1, type=int, help="0: Pretrain模型，1: SFT模型，2: SFT-多图模型 (beta拓展)")
    args = parser.parse_args()

    # 导入模型
    # 首先设置模型的参数
    # 然后再用init_model导入模型
    lm_config = VLMConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=args.use_moe
    )
    model, tokenizer, vision_model, preprocess = init_model(lm_config, args.device)
    # 创建文本流生成器
    # 将模型生成的文字实时显示在屏幕上
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def chat_with_vlm(
            prompt, 
            pixel_values, 
            image_names, 
    ):
        """
        输入prompt和图像，输出模型生成的文字
        """
        # 生成prompt
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )[-args.max_seq_len+1:]

        inputs = tokenizer(
            new_prompt, 
            return_tensors="pt", 
            trunction=True, 
        ).to(args.device)

        print(f"[Image]: {image_names}")
        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len, 
            num_return_sequences=1,  # 只生成一个回答
            do_sample=True, # 启用随机采样，增加结果的多样性
            attention_mask=inputs["attention_mask"], 
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
            streamer=streamer, # 流式输出
            top_p=args.top_p, # 采样参数
            temperature=args.temperature, # 温度参数
            pixel_values=pixel_values, # 图像特征
        )

        # 将生成的回答解码为文本
        response = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        messages.append({
            "role": "assistant", 
            "content": response
        })
        print("\n\n")

    if args.use_multi == 1:
        image_dir = "./dataset/eval_images"
        prompt = f"{model.params.image_special_token}\n描述一下这个图像的内容。"

        for image_file in os.listdir(image_dir):
            # 导入图像
            image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
            # 将图像转化为tensor
            pixel_tensors = MiniMindVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)
            # 调用模型
            chat_with_vlm(prompt, pixel_tensors, image_file)

    if args.use_multi == 2:
        args.model_mode = 2
        image_dir = "./dataset/eval_images/bird/"
        prompt = (
            f"{lm_config.image_special_token}\n"
            f"{lm_config.image_special_token}\n"
            f"比较一下这两张图片，它们有什么异同点？"
        )
        pixel_tensors_multi = []
        for image_file in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
            pixel_tensors_multi.append(MiniMindVLM.image2tensor(image, preprocess))

        pixel_tensors = torch.cat(pixel_tensors_multi, dim=0).to(args.device).unsqueeze(0)

        for _ in range(10):
            chat_with_vlm(prompt, pixel_tensors, (', '.join(os.listdir(image_dir))))