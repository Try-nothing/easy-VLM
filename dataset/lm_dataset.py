import json 
import os 
import torch
from PIL import Image 
from torch.utils.data import Dataset, DataLoader 
from model.model_vlm import MiniMindVLM 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLMDataset(Dataset):
    def __init__(
            self, 
            jsonl_path, 
            images_path, 
            tokenizer, 
            preprocess=None, 
            max_length=512, 
            image_special_token="@"*196, 
    ):
        super().__init__()
        self.samples = self.load_data(jsonl_path) # samples是一个list，每个元素是一条样本
        self.images_path = images_path # 每个样本对应的图片

        self.tokenizer = tokenizer # 文本编码器
        self.max_length = max_length
        self.preprocess = preprocess # 图片预处理函数
        self.image_token = image_special_token 
        self.bos_id = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids 
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids 

    def __len__(self):
        return len(self.samples)

    def load_data(
            self, 
            path, 
    ):
        """ 
        从对应的jsonl文件中加载数据
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f,1):
                # 按行读取jsonl文件
                data = json.loads(line.strip())
                samples.append(data)
        return samples 
    
    def _create_chat_prompt(self, conversations):
        # 一个conversations是一个list，其中包含多轮对话
        # user和assistant交替出现
        messages = [] 
        for i, turn in enumerate(conversations):
            # turn是一个dict，包含role和content
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({
                "role": role, 
                "content": turn['content'].replace('<image>', self.image_token) # 将<image>替换为图片token @
            })
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False, 
        ) # ​​原始对话记录​​ 转换为模型所需的 ​​结构化文本格式
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到<|im_start|>的位置
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start 
                while end < len(input_ids):
                    # 找到<|im_end|>的位置
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break 
                    end += 1 
                # 将<|im_start|>和<|im_end|>之间的token的loss_mask设为1
                for j in range(start+1, min(end+len(self.eos_id)+1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index:int):
        sample = self.samples[index] # 获取第index条样本
        image_paths = sample['image'] # 获取这个样本的图片路径
        prompt = self._create_chat_prompt(sample['conversations']) # 将对话转换为结构化文本
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 将文本转化为token
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) # 如果长度不够，则用pad_token_id填充
        loss_mask = self._generate_loss_mask(input_ids) # 根据<|im_start|>和<|im_end|>生成loss_mask

        X = torch.tensor(input_ids[:-1], dtype=torch.long) # 去掉最后一个token，因为最后一个token是pad_token_id
        Y = torch.tensor(input_ids[1:], dtype=torch.long) # 去掉第一个token，因为第一个token是bos_token_id
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long) # 去掉第一个token，因为第一个token是bos_token_id

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f"{self.images_path}/{image_name}") # 读取图片
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess) # 将图片转化为tensor
            image_tensors.append(image_tensor) # 将图片tensor添加到list中
        image_tensors = torch.stack(image_tensors, dim=0) # 将list中的tensor堆叠成tensor

        return X, Y, loss_mask, image_tensors

