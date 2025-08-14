import os 
import torch 
import warnings 
from .model_minimind import * 
from typing import Optional, Tuple, List 
from torch import nn 
from transformers import CLIPProcessor, CLIPModel 

warnings.filterwarnings("ignore")


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self, 
            image_special_token: str = "@" * 196,
            image_ids: List = [34] * 196,
            **kwargs
    ):
        # 图像占位符
        # "Describe this image: <image_special_token> The scene shows..."
        self.image_special_token = image_special_token 
        # 图像id，用来表明该位置是图片token
        # 在运行中，会将图片的嵌入插入到这些地方
        self.image_ids = image_ids 
        super().__init__(**kwargs)

class VisonProj(nn.Module):
    def __init__(self, ve_hideen_size=768, hidden_size=512):
        super().__init__()
        # CLIP的输出维度
        self.ve_hidden_size = ve_hideen_size
        # 语言模型的隐藏层维度
        self.hidden_size = hidden_size
        # 将图像嵌入映射到语言模型的隐藏层维度
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size),
        )
    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj
        

class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params 
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisonProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path:str):
        """
        用于加载图像编码器和图像处理器
        """
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error() # 禁用transformers日志
        if not os.path.exists(model_path):
            return None, None 
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path) # 将输入处理为encoder可以处理的格式
        for param in model.parameters():
            param.requires_grad = False 
        return model.eval(), processor 
    
    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', "LA"]: image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs
    
    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
            image_embedding = outputs.last_hidden_state[:, 1:,:].squeeze()
        return image_embedding
    
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        # 在token序列中定位图像占位符
        def find_indices(tokens, image_ids):
            # 找到序列中所有等于image_ids的索引
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)

            # 如果image_ids_tensor的长度大于tokens的长度，则返回None
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None 
            
            # 找到所有等于image_ids的索引
            # tensor.unfold(dimension, size, step)
            # dimension: 要展开的维度
            # size: 每个窗口的大小
            # step: 每个窗口的步长
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            if matches.sum() == 0:
                return None
            # 找到所有等于image_ids的索引
            match_indices = {}
            for batch_idx in range(tokens.size(0)):
                # 将batch拆开，一条数据一条数据的遍历
                if matches[batch_idx].any(): # 检查当前样本是否有至少一个匹配窗口
                    # 为每个匹配窗口计算起止位置
                    # matches[batch_idx].nonzero()返回所谓非零元素的位置
                    match_indices[batch_idx] = [
                        (idx.item(), idx.item() + len_image_ids - 1) for idx in matches[batch_idx].nonzero(as_tuple=True)[0]
                    ]
        image_indices = find_indices(tokens, self.params.image_ids) # image_indices[i]中放着第i条数据对应的图片占位符所在的位置

        # 将图像嵌入插入到语言模型的隐藏层
        if vision_tensors is not None and image_indices:
            # 获取图像嵌入
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0) # batch size, num_images, patch_nums, hidden_embeddings

            new_h = []
            for i in range(h.size(0)): # 一条一条的取出数据
                if i in image_indices: # 如果这条数据中存在图片数据
                    h_i = h[i] # 取出文本的隐藏层
                    img_idx = 0 # 一条数据中不一定只有一张图片
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # 将图像嵌入插入到隐藏层中
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx+1:]), dim=0)[:seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
            use_cache: bool = False, 
            logits_to_keep: Union[int, torch.Tensor] = 0, 
            pixel_values: Optional[torch.FloatTensor] = None, 
            **args
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids)) # 获得文本的embeddings

        if pixel_values is not None and start_pos == 0: # 如果有图像，就将图像的embeddings插入到文本的embeddings中
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape 
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder) for i in range(num)
            ], dim=stack_dim) # 获取所有图片的clip嵌入
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1]) # 将图片的embeddings插入到文本的embeddings中

        position_embeddings = (
            self.model.freqs_cos[start_pos: start_pos+seq_length],
            self.model.freqs_sin[start_pos: start_pos+seq_length]
        ) # 获取位置嵌入

        # 运行模型
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT