import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """PQ Codebook模型（用于推理，只用于量化，不训练）"""
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
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, seq_len, output_dim)
        """
        return self.mlp(x)


def load_models(codebook_path, mlp_path, llm_path, device="cuda:0"):
    """加载所有模型"""
    print(f"Loading models on {device}...")
    
    # 加载PQ codebook
    print(f"  Loading PQ codebook from {codebook_path}...")
    pq_codebook_model = PQCodebookModel(codebook_path, device=device)
    pq_codebook_model.eval()
    
    # 加载MLP
    print(f"  Loading MLP from {mlp_path}...")
    mlp_checkpoint = torch.load(mlp_path, map_location=device)
    mlp_model = MLPProjection(
        input_dim=mlp_checkpoint["input_dim"],
        hidden_dim=mlp_checkpoint["hidden_dim"],
        output_dim=mlp_checkpoint["output_dim"]
    )
    mlp_model.load_state_dict(mlp_checkpoint["mlp"])
    mlp_model.to(device)
    mlp_model.eval()
    
    # 加载LLM
    print(f"  Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    
    # 确保chat_template已加载
    ensure_chat_template(tokenizer, llm_path)
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device}  # 只使用指定的device
    )
    llm_model.eval()
    
    # 确保占位符在tokenizer中
    placeholder_token = "<USR_EMB>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        print(f"  Added placeholder token: {placeholder_token}")
    
    print("All models loaded!")
    return pq_codebook_model, mlp_model, llm_model, tokenizer


class InferenceDataset(Dataset):
    """推理数据集类"""
    def __init__(self, jsonl_path, embeddings_path, tokenizer, his_len=8, max_length=500):
        self.tokenizer = tokenizer
        self.his_len = his_len
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        # 加载预计算的embeddings
        print(f"Loading pre-computed embeddings from {embeddings_path}...")
        embeddings_data = torch.load(embeddings_path, map_location='cpu')
        
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
        
        # 占位符token
        self.placeholder_token = "<USR_EMB>"
        if self.placeholder_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([self.placeholder_token])
            print(f"Added placeholder token: {self.placeholder_token}")
    
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


def collate_fn(batch):
    """Collate function for DataLoader"""
    prompts = [item["prompt"] for item in batch]
    chosens = [item["chosen"] for item in batch]
    user_embeddings = torch.stack([item["user_embeddings"] for item in batch])  # (batch_size, his_len, 1024)
    
    return {
        "prompt": prompts,
        "chosen": chosens,
        "user_embeddings": user_embeddings
    }


def inference_batch(batch, pq_codebook_model, mlp_model, llm_model, tokenizer, 
                   his_len=8, max_new_tokens=512, device="cuda:0"):
    """
    对batch进行推理
    
    Args:
        batch: 包含prompt, chosen, user_embeddings的字典
        pq_codebook_model: PQ codebook模型
        mlp_model: MLP投影模型
        llm_model: LLM模型
        tokenizer: LLM的tokenizer
        his_len: 历史长度
        max_new_tokens: 最大生成token数
        device: 设备
    
    Returns:
        generated_texts: 生成的文本列表
    """
    placeholder_token = "<USR_EMB>"
    prompts = batch["prompt"]
    user_embeddings = batch["user_embeddings"].to(device)  # (batch_size, his_len, 1024)
    batch_size = len(prompts)
    
    # PQ量化（不需要梯度）
    with torch.no_grad():
        quantized_embs, _ = pq_codebook_model(user_embeddings)  # (batch_size, his_len, 1024)
    
    # MLP投影到LLM维度（不需要梯度）
    with torch.no_grad():
        llm_embs = mlp_model(quantized_embs)  # (batch_size, his_len, llm_dim)
    
    # 获取LLM的embedding层和数据类型
    llm_embeddings = llm_model.get_input_embeddings()
    model_dtype = next(llm_model.parameters()).dtype
    
    # 确保llm_embs使用正确的数据类型
    llm_embs = llm_embs.to(dtype=model_dtype)
    
    # 构建每个样本的prompt并tokenize
    input_ids_list = []
    placeholder_positions_list = []
    
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    for i in range(batch_size):
        prompt = prompts[i]
        
        # 构建prompt
        placeholder_str = " ".join([placeholder_token] * his_len)
        user_prompt_text = (
            "you are a helpful assistant good at writing response based on a question and user prototype.\n"
            f"The user prototype are {placeholder_str}.\n"
            f"The question is {prompt}\n"
            "Directly output your answer. Write about 80 words."
        )
        
        # 使用chat template
        messages = [
            {"role": "user", "content": user_prompt_text}
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=500
        )
        
        input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)
        input_ids_list.append(input_ids)
        
        # 找到占位符位置
        placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()
        placeholder_positions_list.append(placeholder_positions)
    
    # Padding到相同长度
    max_seq_len = max(ids.size(0) for ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    for input_ids in input_ids_list:
        seq_len = input_ids.size(0)
        padding_length = max_seq_len - seq_len
        
        # Left padding
        padded = torch.cat([
            torch.full((padding_length,), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, dtype=input_ids.dtype),
            input_ids
        ])
        padded_input_ids.append(padded)
        
        # Attention mask
        attention_mask = torch.cat([
            torch.zeros(padding_length, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long)
        ])
        attention_masks.append(attention_mask)
    
    input_ids_batch = torch.stack(padded_input_ids).to(device)  # (batch_size, max_seq_len)
    attention_mask_batch = torch.stack(attention_masks).to(device)  # (batch_size, max_seq_len)
    
    # 获取base embeddings
    input_embs = llm_embeddings(input_ids_batch)  # (batch_size, max_seq_len, llm_dim)
    input_embs = input_embs.to(dtype=model_dtype)
    
    # 替换占位符位置的embeddings
    for i in range(batch_size):
        placeholder_positions = placeholder_positions_list[i]
        if len(placeholder_positions) >= his_len:
            # 调整位置（因为left padding）
            padding_length = max_seq_len - input_ids_list[i].size(0)
            adjusted_positions = [pos + padding_length for pos in placeholder_positions[:his_len]]
            
            for j, pos in enumerate(adjusted_positions):
                if pos < max_seq_len:
                    input_embs[i, pos] = llm_embs[i, j]
    
    # 批量生成
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=input_embs,
            attention_mask=attention_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码每个样本（只取新生成的部分）
    generated_texts = []
    for i in range(batch_size):
        input_seq_len = input_ids_list[i].size(0)
        generated_ids = outputs[i]  # 只取新生成的部分
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="Stage 2推理：使用训练好的PQ codebook和MLP")
    
    # 路径参数
    parser.add_argument("--data_path", type=str,
                       default="/mnt/workspace/wangliang/alignx/AlignX_val_1500_unique_prompt_add.jsonl",
                       help="输入的JSONL数据文件路径")
    parser.add_argument("--embeddings_path", type=str,
                       default="/mnt/workspace/wangliang/alignx/AlignX_val_1500_ugc_all_emb.pt",
                       help="预计算的embeddings文件路径")
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="训练好的PQ codebook模型路径")
    parser.add_argument("--mlp_path", type=str, required=True,
                       help="训练好的MLP模型路径")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="LLM模型路径")
    parser.add_argument("--output_path", type=str,
                       default="/mnt/workspace/wangliang/alignx/stage2_inference_results_4.jsonl",
                       help="输出结果文件路径")
    
    # 推理参数
    parser.add_argument("--his_len", type=int, default=4,
                       help="Number of historical Q&A pairs to use")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:2",
                       help="设备 (cuda:0, cpu, etc.)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="处理的样本数量（None表示处理全部）")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers for DataLoader")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    pq_codebook_model, mlp_model, llm_model, tokenizer = load_models(
        codebook_path=args.codebook_path,
        mlp_path=args.mlp_path,
        llm_path=args.llm_path,
        device=device
    )
    
    # 创建数据集和DataLoader
    print(f"\nCreating dataset and dataloader...")
    dataset = InferenceDataset(
        jsonl_path=args.data_path,
        embeddings_path=args.embeddings_path,
        tokenizer=tokenizer,
        his_len=args.his_len,
        max_length=500
    )
    
    # 如果指定了num_samples，只取前N个
    if args.num_samples:
        dataset.data = dataset.data[:args.num_samples]
        print(f"Processing first {len(dataset)} samples")
    else:
        print(f"Processing all {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # 推理
    print(f"\nStarting batch inference (batch_size={args.batch_size})...")
    results = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferencing")):
        try:
            generated_texts = inference_batch(
                batch=batch,
                pq_codebook_model=pq_codebook_model,
                mlp_model=mlp_model,
                llm_model=llm_model,
                tokenizer=tokenizer,
                his_len=args.his_len,
                max_new_tokens=args.max_new_tokens,
                device=device
            )
            
            # 保存结果
            for i, generated_text in enumerate(generated_texts):
                global_idx = batch_idx * args.batch_size + i
                if global_idx < len(dataset):
                    results.append({
                        "index": global_idx,
                        "prompt": batch["prompt"][i],
                        "chosen": batch["chosen"][i],
                        "generated": generated_text
                    })
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # 为这个batch的所有样本添加错误标记
            for i in range(len(batch["prompt"])):
                global_idx = batch_idx * args.batch_size + i
                if global_idx < len(dataset):
                    results.append({
                        "index": global_idx,
                        "prompt": batch["prompt"][i],
                        "chosen": batch["chosen"][i],
                        "generated": f"ERROR: {str(e)}"
                    })
    
    # 保存结果
    print(f"\nSaving results to {args.output_path}...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(results)} results to {args.output_path}")
    
    # 打印一些统计信息
    print(f"\nInference completed!")
    print(f"  Total samples processed: {len(results)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  History length: {args.his_len}")
    print(f"  Max new tokens: {args.max_new_tokens}")


if __name__ == "__main__":
    main()
