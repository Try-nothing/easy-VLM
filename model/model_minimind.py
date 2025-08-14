from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0, 
            bos_token_id: int = 1, 
            eos_token_id: int = 2, 
            hidden_act: str = "silu", 
            hidden_size: int = 512, 
            intermediate_size: int = None, 
            max_position_embeddings: int = 32768, 
            num_attention_heads: int = 8, 
            num_hidden_layers: int = 8, 
            num_key_value_heads: int = 2, 
            vocab_size: int = 6400, 
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0, 
            flash_attn: bool = True, 

            use_moe: bool = False, 
            num_experts_per_tok: int=1,
            n_routed_experts: int=4, 
            n_shared_experts: int=1,
            scoring_func: str='softmax',
            aux_loss_alpha: float=0.1, 
            seq_aux: bool = True, 
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dropout = dropout # 全连接层的Dropout概率
        self.bos_token_id = bos_token_id # 句子开始符
        self.eos_token_id = eos_token_id # 句子结束符
        self.hidden_act = hidden_act # 隐藏层的激活函数
        self.hidden_size = hidden_size # 隐藏层大小
        self.intermediate_size = intermediate_size # 中间层大小
        self.max_position_embeddings = max_position_embeddings # 最大序列长度
        self.num_attention_heads = num_attention_heads # 注意力头的数量
        self.num_hidden_layers = num_hidden_layers # 隐藏层的数量
        self.num_key_value_heads = num_key_value_heads # 键值头的数量
        self.vocab_size = vocab_size # 词汇表大小
        self.rms_norm_eps = rms_norm_eps # RMSNorm的epsilon值
        self.rope_theta = rope_theta # 旋转位置编码的theta值
        self.flash_attn = flash_attn # 是否使用FlashAttention

        self.use_moe = use_moe # 是否使用MoE
        self.num_experts_per_tok = num_experts_per_tok # 每个token的专家数量
        self.n_routed_experts = n_routed_experts # 路由的专家数量
        self.n_shared_experts = n_shared_experts # 共享的专家数量
        self.scoring_func = scoring_func  # 专家选择函数
        self.aux_loss_alpha = aux_loss_alpha # 辅助损失权重
        self.seq_aux = seq_aux # 是否使用序列辅助损失
        self.norm_topk_prob = norm_topk_prob # 是否使用TopK概率归一化


import math 
import torch 
from torch import nn 
# 根据配置文件中指定的激活函数名称获得实际激活函数
# hidden_act: str = "silu",
# hidden_act = ACT2FN[hidden_act]
from transformers.activations import ACT2FN 
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F 
# PretrainedConfig: 用于加载模型配置文件
# PreTrainedModel: 用于加载模型
# GenerationMixin: 用于生成文本，支持多种解码策略：贪心搜索，Beam search，采样，对比搜索
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
# MoeCausalLMOutputWithPast：用于MoE输出模型结果
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # rsqrt是计算倒平方根
        # RMS假设输入的均值为0，那么方差就为x^2的均值
        # eps的目的是防止除以0
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # 归一化后再经过一个线性层
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int=int(32*1024), theta: float=1e6):
    # 1. 计算频率向量 (维度: [dim//2])
    # 计算基础频率 θ_i = 1 / (theta^{2i/dim})
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].float()/dim))

    # 2. 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)

    # 3. 计算外积：位置×频率 -> [end, dim//2]
    # 公式：ω_m = m * θ_i
    freqs = torch.outer(t, freqs).float()

    # 4. 计算对应的余弦和正弦值
    # 注意：由于特征维度分组成对，需要每个值重复两次
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        """
        旋转向量后半部分（实现90度旋转效果）并于前半部分交换
        """
        half_dim = x.shape[-1] // 2
        return torch.cat(
            (
                -x[...,half_dim:], # 符号反转向量的后半部分
                x[...,:half_dim] # 向量的前半部分
            ),
            dim=-1
        )
    
    # 调整位置编码维度（添加空维度以便广播）
    # 例如：cos形状[L,D] -> [L,1,D] 使能与q/k的[BS,L,H,D]相乘
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 应用旋转位置编码到查询向量(Q)
    # 数学公式：q_rot = q * cos(位置) + rotate_half(q) * sin(位置)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int):
    bs, slen, num_key_value_heads, head_dim = x.shape 
    if n_rep == 1:
        return x
    return (x[:, :, None, :].expand(bs, slen, n_rep, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads*n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 如果num_key_value_heads不为空，则使用num_key_value_heads，否则使用num_attention_heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads # Q的注意力头数
        self.n_local_kv_heads = self.num_key_value_heads # K和V的注意力头数
        # 在设置的时候，Q的注意力头数必须是K和V的注意力头数的整数倍
        # 这里要计算n_rep，用repeat_kv来重复K和V
        # 这样在attention计算的时候，Q和K、V的形状是相同的
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 
        # 计算每个注意力头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.hidden_size * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    def forward(
            self, 
            x: torch.Tensor, 
            position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
            use_cache=False, 
            attention_mask: Optional[torch.Tensor]=None
    ):
        bsz, seq_len, _ = x.shape 
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 如果past_key_value不为空，则将xk和xv与past_key_value拼接
        # 这里是在处理KV缓存 
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 保存kv缓存
        past_kv = (xk, xv) if use_cache else None 

        # 重复KV
        xq, xk, xv = (
            xq.transpose(1,2),
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2)
        )

        # 如果有flash attn，就使用flash attn
        # 如果没有flash attn，就使用标准的attention
        if self.flash and seq_len != 1: # Flash Attention仅支持非单点序列。当序列长度为1时，使用标准的attention。
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                # 将mask重塑为Flash Attention要求的格式 [batch_size, num_heads, q_len, k_len]
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
            attn_mask = attn_mask.bool() if attention_mask is not None else None
            
            # 调用Flash Attention（高效优化）
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True
            )
        else:
            # 计算注意力得分
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 创建一个全-inf的矩阵，然后保留下三角
            # 然后reshape成与scores相同的形状
            # 再与scores相加
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1 # 对角线为1，即下三角
            ).unsqueeze(0).unsqueeze(0)

            # 添加额外的注意力掩码
            if attention_mask is not None:
                # 扩展掩码维度 [B, L] -> [B, 1, 1, L]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 转换: 1表示保留,0表示掩码
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            # 计算softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        # 调整维度: [B, H, L, D] -> [B, L, H*D]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        实现SwiGLU
        """
        super().__init__()
        # 计算中间层维度
        if config.intermediate_size is None:
            # 计算默认中间层大小：8/3 ≈ 2.67倍于隐藏层维度
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 确保中间层大小是64的倍数（内存对齐优化）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        为每个token分配专家
        应用一个线性层，做分类，分类为每个token对应的专家
        使用负载均衡损失，防止某些专家过载而其他专家闲置
        """
        super().__init__()
        self.config = config 
        self.top_k = config.num_experts_per_tok # 每个token选择调用的专家数量（通常1-2）
        self.n_routed_experts = config.n_routed_experts # 可选专家总数

        self.scoring_func = config.scoring_func # 打分函数（默认softmax）
        self.alpha = config.aux_loss_alpha  # 辅助损失权重系数
        self.seq_aux = config.seq_aux # 序列级辅助损失开关，如果为True，则使用负载均衡损失，损失在总的损失中的权重就为alpha

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size # 输入特征维度
        self.weight =nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, hidden_size = hidden_states.shape 
        hidden_states = hidden_states.view(-1, hidden_size)
        # 分类
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring func for MoE gating: {self.scoring_func}")
        # 为每个token选择前k个专家
        # topk_idx: 被选中的专家索引 [batch*seq, top_k]
        # topk_weight: 对应专家的权重分数 [batch*seq, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果每个token的专家数大于1，则需要对权重进行归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores 
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # 此处可以看做是在统计每个句子中分配到的专家数量
            if self.seq_aux: # 启用序列级负载损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # 重组得分矩阵 [batch, seq_len, n_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 统计每个batch中每个专家被选中的次数
                # scatter_add_函数根据topk_idx中的索引，将torch.ones中的1累加到ce的对应位置
                # ce: [batch, n_experts]
                # topk_idx: [batch*seq_len, top_k] -> [batch, seq_len*top_k]
                # torch.ones: [batch, seq_len*top_k]
                # ce[topk_idx[i,j]] += torch.ones[i,j]
                # torch.ones表示要加的值，topk_idx表示要加到的位置
                ce.scatter_add_(
                    1,  # 沿着专家维度统计
                    topk_idx.view(bsz, -1),  # 展平的专家索引
                    torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device)
                )
                # 归一化专家负载，防止seq_len * top_k过大导致损失值过大
                # 乘以n_routed_experts是为了保证损失值在合理范围内
                ce.div_(seq_len * self.top_k / self.n_routed_experts)
                # 计算辅助损失：负载分布与门控分数的差异
                # scores_for_seq_aux.mean(dim=1)计算每个专家的平均权重
                # ce * scores_for_seq_aux.mean(dim=1)计算在一个 batch 内，某个专家在所有 token 上的平均门控分数​
                # (ce*scores_for_seq_aux.mean(dim=1)).sum(dim=1)计算每个batch的加权负载损失

                # ce为每个专家出现的实际频率
                # scores_for_seq_aux.mean(dim=1)为每个专家的期望概率
                # 根据柯西不等式：sum_{i=1}^{N}(x_i * y_i) ** 2  <= (sum_{i=1}^{N}x_i ** 2) * (sum_{i=1}^{N}y_i ** 2)
                # 当且仅当 x_i = y_i 时等号成立
                # 而x_i=y_i=1/N时取值达到最小
                # 所以ce和scores_for_seq_aux.mean(dim=1)的乘积越小，损失越小
                aux_loss = (ce*scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha

                # 每个专家的实际使用次数乘以实际使用频率就是期望
            else: # 计算token级负载损失
                # mask_ce: 若第i个token第j个expert被选中，则mask_ce[i,j] = 1，否则为0
                # ce: 每个专家的实际使用频率
                # Pi: 每个专家的期望使用频率
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0.0
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config 

        # 定义路由专家
        # 路由专家会用一个MLP来计算每个token应该被分配到哪个专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)

        # 定义共享专家
        # 共享专家会用在所有token上，不需要分配
        if config.n_shared_experts > 0:
            self.n_shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
    
    def forward(self, x:torch.Tensor):
        # 复制输入
        identity = x 
        orig_shape = x.shape 
        bsz, seq_len, _ = x.shape 

        # 获得路由专家分配
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 将输入展开为token形式[batch*seq, hidden]
        x = x.view(-1, x.shape[-1])
        # 将专家分配转化为单token模式[batch*seq*top_k]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 将输入复制top_k次，以便每个token都有top_k个专家
            # 比如top_k=2, 则需要将x复制为[batch*seq*2, hidden]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y的形状与x相同
            y = torch.empty_like(x, dtype=torch.float16)

            # 对每个专家，计算分配给它的token
            # flat_topk_idx中储存的就是每个token被分配到的专家
            # 只需要用flat_topk_idx==i就可以定位到对应的token
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx==i] = expert(x[flat_topk_idx==i]).to(y.dtype)
            # 根据topk_weight对每个token进行加权
            # y的形状为[batch*seq*top_k, hidden]
            # topk_weight的形状为[batch*seq, top_k]
            # y: [batch*seq*top_k, hidden] -> [batch*seq, top_k, hidden] -> [batch*seq, hidden] -> [batch, seq, hidden]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 将y的形状恢复为[batch, seq, hidden]
            y = y.view(*orig_shape)
        else:
            # 非训练模式下，直接计算每个token的输出
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 调用共享专家
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x: [batch*seq, hidden]
        expert_cache = torch.zeros_like(x)
        # 按专家索引排序
        # idxs[i] = j代表排序后序列号为i的专家的原始序列号为j，也就是flat_expert_indices[j]第i大
        idxs = flat_expert_indices.argsort()
        # 计算每个专家需要处理的token数
        # bincount统计每个专家出现的次数
        # cumsum计算累积和，计算每个专家处理的token在排序中的结束位置
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 从专家索引逆推token索引
        # token总共有batch*seq个，而专家共有batch*seq*top_k个
        # 也就是说专家数为token数的topk倍
        # idxs // top_k就是token索引
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            # 获得该专家对应的token在张量中索引的起始位置和结束位置
            # 这样就可以得到该专家需要处理的token
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            # 获取专家
            expert = self.experts[i]
            # 获取token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 根据token索引获取token
            expert_tokens = x[exp_token_idx]
            # 计算专家的输出
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 根据权重对专家的输出进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将专家的输出累加到expert_cache中
            # exp_token_idx: [batch*seq] -> [batch*seq, 1] -> [batch*seq, hidden]
            expert_cache.scatter_add_(0, exp_token_idx.view(-1,1).repeat(1,x.shape[-1]), expert_out)

        return expert_cache

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size 
        self.head_dim = config.hidden_size // config.num_attention_heads 
        self.self_attn = Attention(config)

        self.layer_id = layer_id 
        # attn前进行一次归一化
        # attn后进行一次归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
            self, 
            hidden_states, 
            position_embeddings, 
            past_key_value=None, 
            use_cache=False, 
            attention_mask=None
            ):
        # 复制输入
        residual = hidden_states 
        # 应用注意力模块
        # 先对输入进行layernorm，然后计算注意力
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value, 
            use_cache, 
            attention_mask
        )
        # 残差链接
        hidden_states += residual 
        # 再进行一个layer_norm
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config=config 
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(l,config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算旋转位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size// config.num_attention_heads, end=config.max_position_embeddings, theta=config.rope_theta)
        
        # 将预计算的旋转位置编码注册为模型参数，以便在训练过程中使用
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
            use_cache: bool = False, 
            **kwargs
    ):
        # 获取输入序列的batch大小和序列长度
        batch_size, seq_length = input_ids.shape 
        past_key_values = past_key_values or [None] * len(self.layers)
        # 通过KV cache计算序列的其实位置
        # 在KV cache中，Q是单独的一个token，无法计算这个token在序列当中的位置
        # 但是KV cache是完整的序列，所以可以通过KV cache的长度来计算序列的起始位置
        # 起始位置就为KV cache的长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 

        # 将输入序列的嵌入向量通过Dropout层
        # 这是为了防止过拟合，通过Dropout层可以随机地丢弃一部分神经元，从而降低模型的复杂度
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 计算旋转位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos+seq_length],
            self.freqs_sin[start_pos:start_pos+seq_length]
        )

        # 前向传播，并记录本次前向传播的KV cache
        presents = []
        for layer_idx, (layer, past_key_values) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, presents = layer(
                hidden_states, 
                position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(presents)

        hidden_states = self.norm(hidden_states)

        # 计算MoE负载均衡辅助损失。因为在每一层中都存在MoE结构，所有要将每层的辅助损失都加起来
        aux_loss = sum(
            layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward)
        )

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig=None):
        self.config = config or MiniMindConfig() # 加载配置文件,如果没有指定配置文件,则使用默认配置文件

        super().__init__(self.config)
        self.model = MiniMindModel(self.config) # 加载基础模型
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) # 加载线性分类层
        # 将嵌入层的权重与线性分类层的权重绑定在一起, 以避免重复计算。
        # 这是因为希望将嵌入层的权重与线性分类层的权重绑定在一起，以避免重复计算。
        # 这样做的好处是可以减少内存占用，并且可以加快训练速度。
        self.model.embed_tokens.weight = self.lm_head.weight 
        self.OUT = MoeCausalLMOutputWithPast() # 加载输出层。用于规范化输出，并计算损失函数。

    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False, 
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **args
    ):
        # 调用模型（返回隐藏状态、缓存和辅助损失）
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 计算需要保留logits的索引位置（显存优化）
        # 如果logits_to_keep是整数，则保留最后N个token的logits
        # 如果是张量，则直接使用指定的索引
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT 