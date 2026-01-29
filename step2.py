import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import numpy as np


def ensure_chat_template(tokenizer, model_path):
    """
    确保tokenizer有chat_template，如果没有则从文件中加载
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        print(f"✅ tokenizer已有 chat_template")
        return
    
    # 尝试从文件中加载chat_template
    for name in ("chat_template.jinja", "chat_template.txt", "chat_template.json"):
        template_path = os.path.join(model_path, name)
        if os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                print(f"✅ 从文件加载 chat_template: {template_path}")
                return
            except Exception as e:
                print(f"⚠️  读取 chat_template 失败: {template_path} -> {e}")
    
    print(f"⚠️  未找到 chat_template 文件，请确保 {model_path} 目录下有 chat_template.jinja")


class PQCodebookModel(nn.Module):
    """PQ Codebook模型（用于Stage 2，只用于量化，不训练）"""
    def __init__(self, codebook_path, device="cpu"):
        super().__init__()
        # 加载训练好的PQ codebook
        checkpoint = torch.load(codebook_path, map_location=device)
        
        if "codebooks" in checkpoint:
            # 将codebooks转换为ParameterList
            codebooks_list = []
            for cb in checkpoint["codebooks"]:
                if isinstance(cb, torch.Tensor):
                    codebooks_list.append(nn.Parameter(cb.to(device), requires_grad=False))
                else:
                    codebooks_list.append(nn.Parameter(torch.tensor(cb, device=device), requires_grad=False))
            
            self.codebooks = nn.ParameterList(codebooks_list)
            self.num_subspaces = checkpoint.get("num_subspaces", len(self.codebooks))
            self.subspace_dim = checkpoint.get("subspace_dim", self.codebooks[0].shape[1] if len(self.codebooks) > 0 else None)
            self.codebook_size = self.codebooks[0].shape[0]
            self.emb_dim = self.num_subspaces * self.subspace_dim
        else:
            raise ValueError(f"Checkpoint must contain 'codebooks' key. Found keys: {checkpoint.keys()}")
        
        print(f"Loaded PQ codebook: {self.num_subspaces} subspaces, each {self.subspace_dim}D, {self.codebook_size} entries per subspace")
    
    def quantize(self, embeddings):
        """
        Product Quantization: 将embeddings分成多个子空间，每个子空间独立量化
        embeddings: (batch, seq_len, emb_dim)
        返回: (batch, seq_len, emb_dim) 量化后的embeddings, (batch, seq_len, num_subspaces) 每个子空间的索引
        """
        batch_size, seq_len, emb_dim = embeddings.shape
        flat_embs = embeddings.view(-1, emb_dim)  # (batch * seq_len, emb_dim)
        
        # 将embeddings分成子空间
        subspace_embs = flat_embs.view(-1, self.num_subspaces, self.subspace_dim)
        
        quantized_parts = []
        all_indices = []
        
        # 对每个子空间独立量化
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]  # (batch * seq_len, subspace_dim)
            
            # 确保codebook在正确的设备上
            codebook = codebook.to(subspace.device)
            
            # 计算距离: (batch * seq_len, codebook_size)
            distances = torch.cdist(subspace, codebook, p=2)
            
            # 找到最近邻索引
            indices = torch.argmin(distances, dim=-1)  # (batch * seq_len,)
            
            # 从codebook中获取量化后的embeddings
            quantized = codebook[indices]  # (batch * seq_len, subspace_dim)
            
            quantized_parts.append(quantized)
            all_indices.append(indices)
        
        # 拼接所有子空间的量化结果
        quantized = torch.cat(quantized_parts, dim=-1)  # (batch * seq_len, emb_dim)
        quantized = quantized.view(batch_size, seq_len, emb_dim)
        
        # 所有子空间的索引: (batch * seq_len, num_subspaces)
        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = all_indices.view(batch_size, seq_len, self.num_subspaces)
        
        return quantized, all_indices
    
    def forward(self, embeddings):
        """量化embeddings（不训练，只用于推理）"""
        quantized, indices = self.quantize(embeddings)
        return quantized, indices


class MLPProjection(nn.Module):
    """MLP投影层：将量化后的embeddings投影到LLM维度"""
    def __init__(self, input_dim=1024, hidden_dim=None, output_dim=4096):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim  # 默认使用output_dim作为hidden_dim
        
        # 两层MLP: input_dim -> hidden_dim -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        print(f"Initialized MLP: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, seq_len, output_dim)
        """
        return self.mlp(x)


class Stage2Dataset(Dataset):
    """Stage 2数据集：从JSONL读取，使用预计算的embeddings"""
    def __init__(self, jsonl_path, embeddings_path, tokenizer, his_len=8, max_length=2048, device="cpu"):
        self.tokenizer = tokenizer
        self.his_len = his_len
        self.max_length = max_length
        self.device = device
        
        # 加载数据
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        # 加载预计算的embeddings
        print(f"Loading pre-computed embeddings from {embeddings_path}...")
        embeddings_data = torch.load(embeddings_path, map_location='cpu')
        
        # 处理不同的数据格式
        if isinstance(embeddings_data, torch.Tensor):
            if embeddings_data.dim() == 2:
                self.embeddings = embeddings_data  # (N, 1024)
            else:
                raise ValueError(f"Expected 2D tensor (N, 1024), got shape {embeddings_data.shape}")
        elif isinstance(embeddings_data, list):
            self.embeddings = torch.stack(embeddings_data)
        else:
            raise ValueError(f"Unknown embeddings format: {type(embeddings_data)}")
        
        print(f"Loaded {len(self.data)} data samples")
        print(f"Loaded {len(self.embeddings)} pre-computed embeddings")
        print(f"Embedding shape: {self.embeddings.shape}")
        
        # 占位符token
        self.placeholder_token = "<USR_EMB>"
        # 确保占位符在LLM的tokenizer词表中
        if self.placeholder_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([self.placeholder_token])
            print(f"Added placeholder token to LLM tokenizer: {self.placeholder_token}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        user_content = item.get("User-Generated Content", [])
        
        # 从预计算的embeddings中获取对应的embeddings
        user_embs_list = []
        for i, content_item in enumerate(user_content[:self.his_len]):
            order = content_item.get("order")
            if order is not None and isinstance(order, int) and 0 <= order < len(self.embeddings):
                # 使用order索引获取对应的embedding
                emb = self.embeddings[order]
                user_embs_list.append(emb)
            else:
                user_embs_list.append(torch.zeros(1024))
        
        # 如果不够，用零向量填充
        while len(user_embs_list) < self.his_len:
            user_embs_list.append(torch.zeros(1024))
        
        # 堆叠成tensor: (his_len, 1024)
        user_embs = torch.stack(user_embs_list[:self.his_len])
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "user_embeddings": user_embs  # (his_len, 1024)
        }


class Stage2Trainer:
    """Stage 2训练器：训练MLP，LLM冻结"""
    def __init__(self, llm_model, mlp_model, pq_codebook_model, tokenizer, accelerator, his_len=8, max_length=2048):
        self.llm_model = llm_model
        self.mlp_model = mlp_model
        self.pq_codebook_model = pq_codebook_model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.his_len = his_len
        self.max_length = max_length
        self.placeholder_token = "<USR_EMB>"
        
        # LLM冻结
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # PQ codebook冻结（已经是requires_grad=False）
        # MLP需要训练
        for param in self.mlp_model.parameters():
            param.requires_grad = True
    
    def compute_loss(self, batch):
        """计算loss"""
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        user_embeddings = batch["user_embeddings"]  # (batch_size, his_len, 1024)
        
        device = user_embeddings.device
        batch_size = len(prompts)
        his_len = user_embeddings.size(1)
        
        # 对user_embeddings进行PQ量化（不需要梯度）
        with torch.no_grad():
            quantized_embs, _ = self.pq_codebook_model(user_embeddings)  # (batch, his_len, 1024)
        
        # 通过MLP投影到LLM维度（需要梯度）
        llm_embs = self.mlp_model(quantized_embs)  # (batch, his_len, llm_dim)
        
        # 构建输入
        input_ids_list = []
        labels_list = []
        placeholder_positions_list = []
        
        placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        
        for i in range(batch_size):
            user_prompt = prompts[i]
            chosen = chosens[i]
            
            # 构建user prompt，包含his_len个占位符
            placeholder_str = " ".join([self.placeholder_token] * his_len)
            user_prompt_text = (
                "you are a helpful assistant good at writing response based on a question and user prototype.\n"
                f"The user prototype are {placeholder_str}.\n"
                f"The question is {user_prompt}\n"
                "Directly output your answer."
            )
            
            # 使用chat template
            messages = [
                {"role": "user", "content": user_prompt_text},
                {"role": "assistant", "content": chosen}
            ]
            
            # 应用chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize
            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
            
            # 找到占位符的位置
            placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()
            placeholder_positions_list.append(placeholder_positions)
            
            # 创建labels（只计算chosen部分的loss）
            labels = input_ids.clone()
            
            # 找到assistant回复的起始位置
            # 对于Qwen格式，通常是"<|im_start|>assistant\n"之后
            assistant_start = None
            for j in range(len(input_ids) - 1):
                # 查找assistant标记
                if input_ids[j].item() == self.tokenizer.convert_tokens_to_ids("<|im_start|>"):
                    if j + 1 < len(input_ids):
                        # 检查下一个token是否是assistant
                        next_token = input_ids[j + 1].item()
                        # 可能是assistant的token id
                        if "assistant" in self.tokenizer.convert_ids_to_tokens([next_token])[0].lower():
                            assistant_start = j + 3  # 跳过<|im_start|>assistant\n
                            break
            
            if assistant_start is None:
                # 如果找不到，尝试其他方法：找到最后一个占位符之后的位置
                if len(placeholder_positions) > 0:
                    assistant_start = placeholder_positions[-1] + 1
                else:
                    assistant_start = len(input_ids) // 2
            
            # 只计算assistant回复部分的loss
            labels[:assistant_start] = -100
            
            # 占位符位置也不计算loss
            for pos in placeholder_positions:
                labels[pos] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Padding（left padding，因为LLM是decoder-only）
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            # Left padding
            padded_input_ids.append(F.pad(input_ids, (pad_len, 0), value=self.tokenizer.pad_token_id))
            padded_labels.append(F.pad(labels, (pad_len, 0), value=-100))
            attention_masks.append(F.pad(torch.ones(len(input_ids), dtype=torch.long), (pad_len, 0), value=0))
        
        input_ids = torch.stack(padded_input_ids).to(device)  # (batch, max_len)
        labels = torch.stack(padded_labels).to(device)  # (batch, max_len)
        attention_mask = torch.stack(attention_masks).to(device)  # (batch, max_len)
        
        # 调整placeholder_positions（因为left padding）
        adjusted_placeholder_positions = []
        for i, (orig_positions, input_ids_seq) in enumerate(zip(placeholder_positions_list, input_ids_list)):
            pad_len = max_len - len(input_ids_seq)
            adjusted = [pos + pad_len for pos in orig_positions]
            adjusted_placeholder_positions.append(adjusted)
        
        # 获取LLM的embedding层
        llm_embeddings = self.llm_model.get_input_embeddings()
        input_embs = llm_embeddings(input_ids)  # (batch, max_len, llm_dim)
        
        # 替换占位符位置的embeddings
        for i in range(batch_size):
            placeholder_positions = adjusted_placeholder_positions[i]
            if len(placeholder_positions) >= his_len:
                # 将llm_embs插入到占位符位置
                for j, pos in enumerate(placeholder_positions[:his_len]):
                    input_embs[i, pos] = llm_embs[i, j]
        
        # 前向传播
        outputs = self.llm_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        return loss


def train(args):
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print(f"Stage 2: Training MLP projection (LLM frozen)")
        print(f"History length: {args.his_len}")
    
    # 加载LLM模型
    print(f"Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    
    # 确保chat_template已加载
    ensure_chat_template(tokenizer, args.llm_path)
    
    # 设置padding side（decoder-only模型使用left padding）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None  # 使用accelerate管理设备
    )
    
    # 获取LLM的hidden size
    llm_dim = llm_model.config.hidden_size
    print(f"LLM hidden size: {llm_dim}")
    
    # 加载PQ codebook（冻结）
    print(f"Loading PQ codebook from {args.codebook_path}...")
    # 先加载到CPU，后续会通过accelerate移动到正确设备
    pq_codebook_model = PQCodebookModel(args.codebook_path, device="cpu")
    for param in pq_codebook_model.parameters():
        param.requires_grad = False
    
    print(f"PQ codebook info: {pq_codebook_model.num_subspaces} subspaces, {pq_codebook_model.emb_dim}D input")
    
    # 创建MLP投影层（使用LLM的实际维度）
    hidden_dim = args.hidden_dim if args.hidden_dim else llm_dim
    mlp_model = MLPProjection(
        input_dim=pq_codebook_model.emb_dim,  # 使用codebook的实际输入维度
        hidden_dim=hidden_dim,
        output_dim=llm_dim  # 使用LLM的实际维度
    )
    
    if accelerator.is_local_main_process:
        print(f"MLP: {pq_codebook_model.emb_dim} -> {hidden_dim} -> {llm_dim}")
    
    # 创建数据集（使用预计算的embeddings）
    dataset = Stage2Dataset(
        jsonl_path=args.data_path,
        embeddings_path=args.embeddings_path,
        tokenizer=tokenizer,
        his_len=args.his_len,
        max_length=args.max_length,
        device="cpu"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建训练器
    trainer = Stage2Trainer(
        llm_model=llm_model,
        mlp_model=mlp_model,
        pq_codebook_model=pq_codebook_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        his_len=args.his_len,
        max_length=args.max_length
    )
    
    # 创建优化器（只优化MLP）
    optimizer = torch.optim.AdamW(
        mlp_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 使用accelerate准备
    mlp_model, optimizer, dataloader = accelerator.prepare(
        mlp_model, optimizer, dataloader
    )
    
    # 将PQ codebook移到accelerator管理的设备上
    pq_codebook_model = pq_codebook_model.to(accelerator.device)
    
    # 将LLM移到accelerator管理的设备上（如果需要）
    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    
    # 更新trainer中的pq_codebook_model引用
    trainer.pq_codebook_model = pq_codebook_model
    
    # 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss = trainer.compute_loss(batch)
            
            accelerator.backward(loss)
            
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(mlp_model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            global_step += 1
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}"
                })
            
            # 保存checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_local_main_process:
                    save_path = os.path.join(args.output_dir, f"stage2_checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_mlp = accelerator.unwrap_model(mlp_model)
                    # 获取实际的维度（从MLP模型）
                    actual_input_dim = unwrapped_mlp.mlp[0].in_features
                    actual_hidden_dim = unwrapped_mlp.mlp[0].out_features
                    actual_output_dim = unwrapped_mlp.mlp[-1].out_features
                    checkpoint_dict = {
                        "mlp": unwrapped_mlp.state_dict(),
                        "step": global_step,
                        "input_dim": actual_input_dim,
                        "hidden_dim": actual_hidden_dim,
                        "output_dim": actual_output_dim,
                        "his_len": args.his_len
                    }
                    
                    torch.save(checkpoint_dict, os.path.join(save_path, "mlp_model.pt"))
                    print(f"Saved checkpoint to {save_path}")
    
    # 最终保存
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "stage2_final")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_mlp = accelerator.unwrap_model(mlp_model)
        actual_input_dim = unwrapped_mlp.mlp[0].in_features
        actual_hidden_dim = unwrapped_mlp.mlp[0].out_features
        actual_output_dim = unwrapped_mlp.mlp[-1].out_features
        final_dict = {
            "mlp": unwrapped_mlp.state_dict(),
            "input_dim": actual_input_dim,
            "hidden_dim": actual_hidden_dim,
            "output_dim": actual_output_dim,
            "his_len": args.his_len
        }
        
        torch.save(final_dict, os.path.join(final_path, "mlp_model.pt"))
        print(f"Stage 2 training completed! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train MLP projection with frozen LLM and PQ codebook")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to JSONL data file")
    parser.add_argument("--embeddings_path", type=str, required=True,
                       help="Path to pre-computed embeddings .pt file")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="Path to LLM model")
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="Path to trained PQ codebook model (.pt file)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # 模型参数
    parser.add_argument("--his_len", type=int, default=8,
                       help="Number of historical Q&A pairs to use")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension for MLP (default: same as LLM hidden size)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    train(args)
