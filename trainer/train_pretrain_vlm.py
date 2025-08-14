import argparse
import time 
import math 
import warnings

warnings.filterwarnings('ignore')
import os
import sys 
import torch 
import torch.distributed as dist 

__package__ = 'trainer'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import nullcontext 
from torch import optim, nn 
from torch.nn.parallel import DistributedDataParallel 
from torch.utils.data import DataLoader, DistributedSampler 
from transformers import AutoTokenizer, AutoModel 
from model.model_vlm import MiniMindVLM, VLMConfig 
from dataset.lm_dataset import VLMDataset 

def Logger(content):
    """ 
    如果不是分布式训练，就打印日志
    """
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    """
    调整学习率
    """
    return lr * 10 + 0.5 * lr * (1+math.cos(math.pi*current_step/total_steps))

def train_epoch(epoch, wandb):
    """ 
    预训练函数
    """
    # 用交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time() 
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        # 使用动态学习率，随着学习的进行，学习率会逐渐改变
        lr = get_lr(epoch*iter_per_epoch+step, args.epochs*iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

        with ctx: # 自动混合精度学习
            # 前向传播
            res = model(X, pixel_values=pixel_values)
            # 计算损失函数
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 根据mask计算掩码损失，只计算mask中为1的部分的损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加上moe中的辅助损失，用于训练moe
            loss += res.aux_loss
            # 计算梯度累积的损失
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward() # 计算梯度

        # 梯度累积
        # 如果累积了足够梯度，就进行反向传播
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪
            # 防止模型参数训练的不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer) # 更新参数
            scaler.update()

            optimizer.zero_grad(set_to_none=True) # 重置优化器

        if step % args.log_interval == 0:
            # 运行足够步后，记录运行结果
            # 主要是为了更加直观地监督训练情况
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.7f} epoch_time:{}min:'.format(
                    epoch+1, 
                    args.epochs, 
                    step, 
                    iter_per_epoch, 
                    loss.item(), 
                    optimizer.param_groups[-1]['lr'], 
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60 
               )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss, 
                        "lr": optimizer.param_groups[-1]['lr'],
                        "epoch_time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 运行足够步后就保存一下模型
            # 防止训练意外停止导致训练结果丢失
            # 还可以方便继续训练
            model.eval()
            moe_path = "_moe" if model_config.use_moe else ""
            ckp = f"{args.save_dir}/pretrain_vlm_{model_config.hidden_size}{moe_path}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 不保存vision_encoder的参数，因为vision_encoder是固定的，不需要训练
            clean_state_dict = {
                k: v for k,v in state_dict.items() if not k.startswith("vision_encoder.")
            }
            clean_state_dict = {k: v.half() for k,v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            model.train()
            
def init_model(model_config: VLMConfig):
    """
    加载模型
    """
    # 加载文本编码器
    tokenizer = AutoTokenizer.from_pretrained("../model", use_fast=True)
    moe_path = "_moe" if model_config.use_moe else ""

    # 加载纯语言模型的权重
    ckp = f"{args.save_dir}/llm_{model_config.hidden_size}{moe_path}.pth"
    model = MiniMindVLM(model_config, vision_model_path="../model/vision_model/clip-vit-base-patch16")
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 冻结非vision_proj的参数
    for name, param in model.named_parameters():
        if "vision_proj" not in name:
            param.requires_grad = False 
    
    Logger(
        f"VLM可训练参数数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )

    _, preprocess = model.vision_encoder, model.processor 
    return model.to(args.device), tokenizer, preprocess 

def init_distributed_mode():
    if not ddp: return 
    global ddp_local_rank, DEVICE 

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_data.jsonl")
    parser.add_argument("--images_path", type=str, default="../dataset/pretrain_images")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_size", default=8, type=int)
    parser.add_argument("--max_seq_len", default=640, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    args = parser.parse_args()

    model_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_size=args.num_hidden_size,
        max_seq_len=args.max_seq_len,
    )
    max_seq_len = model_config.max_seq_len 
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len 
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-V Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置使用混合精度训练
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1 
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb 

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None 

    # 初始化模型
    model, tokenizer, preprocess = init_model(model_config)

    # 初始化数据集
    train_ds = VLMDataset(
        args.data_path, 
        args.images_path, 
        tokenizer, 
        preprocess=preprocess, 
        image_special_token=model_config.image_special_token, 
        max_seq_len=max_seq_len
    )
    # 如果使用分布式训练，需要把数据集也分别加载到对应的GPU上
    train_sampler = DistributedDataParallel(train_ds) if ddp else None 
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        drop_last=False, 
        shuffle=False, 
        num_workers=args.num_workers, 
        sampler=train_sampler
    )

    # 初始化优化器
    # scaler用于在混合精度训练时对梯度进行精度缩放
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # DistributedDataParallel是分布式中的数据并行，它会在每个GPU上复制一份模型，并在每个GPU上分别计算梯度
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)