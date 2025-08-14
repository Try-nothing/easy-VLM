import os 
import sys 

__package__ = "scripts" 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import warnings 
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM 
from model.model_vlm import MiniMindVLM, VLMConfig 

warnings.filterwarnings('ignore', category=UserWarning)

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    将pytorch模型转化为transformers模型
    """
    VLMConfig.register_for_auto_class()
    MiniMindVLM.register_for_auto_class('AutoModelForCausalLM')

    # 导入模型
    lm_model = MiniMindVLM(lm_config, vision_model_path='../model/vision_model/clip-vit-base-patch16')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 导入权重
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)
    # 计算模型参数量
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数：{model_params / 1e6} 百万 = {model_params / 1e9} B')
    del lm_model.vision_encoder
    # 保存模型
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    print(f'模型已保存：{transformers_path}')

def convert_transformers2torch(transformers_path, torch_path):
    """
    将transformers模型转化为pytorch模型
    """
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f'模型已保存：{torch_path}')


if __name__ == '__main__':
    lm_config = VLMConfig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, use_moe=False)

    torch_path = f"../out/sft_vlm_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    
    transformers_path = '../MiniMind2-V'

    convert_torch2transformers_minimind(torch_path, transformers_path)