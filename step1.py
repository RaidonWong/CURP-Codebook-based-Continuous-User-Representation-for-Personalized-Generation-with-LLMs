import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


def balanced_kmeans_init(embeddings, num_clusters, max_iters=100, device="cuda:0"):
    """
    Balanced K-means初始化codebook（GPU加速版本）
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    embeddings = embeddings.to(device).float()
    N, emb_dim = embeddings.shape
    
    w = N // num_clusters
    
    print(f"Initializing codebook with Balanced K-means on {device}...")
    print(f"  Total embeddings: {N}")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Target size per cluster: {w}")
    
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = embeddings[indices].clone()
    
    prev_assignments = None
    for iter in tqdm(range(max_iters), desc="Balanced K-means iterations"):
        unassigned_mask = torch.ones(N, dtype=torch.bool, device=device)
        assignments = torch.zeros(N, dtype=torch.long, device=device) - 1
        
        for k in tqdm(range(num_clusters), desc=f"Iter {iter+1}: Assigning clusters", leave=False):
            if not unassigned_mask.any():
                break
            
            unassigned_indices = torch.where(unassigned_mask)[0]
            # 确保 unassigned_indices 是1维张量
            if unassigned_indices.dim() == 0:
                unassigned_indices = unassigned_indices.unsqueeze(0)
            
            num_unassigned = unassigned_indices.numel()
            if num_unassigned == 0:
                break
            
            unassigned_embs = embeddings[unassigned_indices]
            distances = torch.cdist(unassigned_embs, centroids[k:k+1], p=2).squeeze()
            
            # 如果只有一个未分配的样本，distances可能是0维，需要处理
            if distances.dim() == 0:
                distances = distances.unsqueeze(0)
            
            sorted_indices = torch.argsort(distances)
            sorted_unassigned_indices = unassigned_indices[sorted_indices]
            
            # 确保 sorted_unassigned_indices 是1维张量
            if sorted_unassigned_indices.dim() == 0:
                sorted_unassigned_indices = sorted_unassigned_indices.unsqueeze(0)
            
            num_to_assign = min(w, sorted_unassigned_indices.numel())
            assigned_indices = sorted_unassigned_indices[:num_to_assign]
            
            assignments[assigned_indices] = k
            unassigned_mask[assigned_indices] = False
            
            if num_to_assign > 0:
                centroids[k] = embeddings[assigned_indices].mean(dim=0)
        
        unassigned_indices = torch.where(unassigned_mask)[0]
        # 确保 unassigned_indices 是1维张量
        if unassigned_indices.dim() == 0:
            unassigned_indices = unassigned_indices.unsqueeze(0)
        
        if unassigned_indices.numel() > 0:
            unassigned_embs = embeddings[unassigned_indices]
            distances_to_all = torch.cdist(unassigned_embs, centroids, p=2)
            nearest_clusters = torch.argmin(distances_to_all, dim=1)
            assignments[unassigned_indices] = nearest_clusters
        
        if prev_assignments is not None:
            if torch.equal(assignments, prev_assignments):
                print(f"  Converged at iteration {iter + 1}")
                break
        
        prev_assignments = assignments.clone()
    
    print("  Updating final centroids...")
    for k in tqdm(range(num_clusters), desc="Updating centroids", leave=False):
        cluster_mask = (assignments == k)
        if cluster_mask.any():
            centroids[k] = embeddings[cluster_mask].mean(dim=0)
    
    print(f"  Balanced K-means initialization completed!")
    return centroids


class CodebookModel(nn.Module):
    """Codebook量化模型（Product Quantization版本：2个子空间独立量化）"""
    def __init__(self, codebook_size=10000, emb_dim=768, num_subspaces=2):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.num_subspaces = num_subspaces
        
        # 计算每个子空间的维度
        assert emb_dim % num_subspaces == 0, f"emb_dim ({emb_dim}) must be divisible by num_subspaces ({num_subspaces})"
        self.subspace_dim = emb_dim // num_subspaces
        
        # 为每个子空间创建独立的codebook
        # 每个codebook: (codebook_size, subspace_dim)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, self.subspace_dim))
            for _ in range(num_subspaces)
        ])
        
        print(f"Initialized PQ codebook: {num_subspaces} subspaces, each {self.subspace_dim}D, {codebook_size} entries per subspace")
        print(f"  Total capacity: {codebook_size}^{num_subspaces} = {codebook_size ** num_subspaces:.2e} combinations")
    
    def quantize(self, embeddings):
        """
        Product Quantization: 将embeddings分成多个子空间，每个子空间独立量化
        embeddings: (batch, seq_len, emb_dim)
        返回: (batch, seq_len, emb_dim) 量化后的embeddings, (batch, seq_len, num_subspaces) 每个子空间的索引
        """
        batch_size, seq_len, emb_dim = embeddings.shape
        flat_embs = embeddings.view(-1, emb_dim)  # (batch * seq_len, emb_dim)
        
        # 将embeddings分成子空间
        # (batch * seq_len, num_subspaces, subspace_dim)
        subspace_embs = flat_embs.view(-1, self.num_subspaces, self.subspace_dim)
        
        quantized_parts = []
        all_indices = []
        
        # 对每个子空间独立量化
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]  # (batch * seq_len, subspace_dim)
            
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
        """
        embeddings: (batch, seq_len, emb_dim)
        返回: (batch, seq_len, emb_dim) 量化后的embeddings, (batch, seq_len, num_subspaces) 每个子空间的索引
        """
        quantized, indices = self.quantize(embeddings)
        return quantized, indices


class CodebookDataset(Dataset):
    """第一阶段数据集：只使用embeddings，不涉及LLM和encoder微调"""
    def __init__(self, embeddings_path, batch_size_per_sample=8, device="cpu"):
        """
        embeddings_path: 预计算的embeddings文件路径 (.pt文件)
        batch_size_per_sample: 每个样本使用多少个embeddings（对应his_len）
        """
        self.batch_size_per_sample = batch_size_per_sample
        self.device = device
        
        # 加载embeddings
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings_data = torch.load(embeddings_path, map_location='cpu')
        
        # 处理不同的数据格式
        if isinstance(embeddings_data, torch.Tensor):
            # 如果是2D tensor (N, 768)，直接使用
            if embeddings_data.dim() == 2:
                self.embeddings = embeddings_data
            else:
                raise ValueError(f"Expected 2D tensor (N, 768), got shape {embeddings_data.shape}")
        elif isinstance(embeddings_data, list):
            # 如果是list，转换为tensor
            self.embeddings = torch.stack(embeddings_data)
        else:
            raise ValueError(f"Unknown embeddings format: {type(embeddings_data)}")
        
        print(f"Loaded {len(self.embeddings)} embeddings, shape: {self.embeddings.shape}")
        
        # 将embeddings分组，每组batch_size_per_sample个
        # 这样可以模拟训练时的batch
        self.num_samples = len(self.embeddings) // batch_size_per_sample
        print(f"Creating {self.num_samples} samples (each with {batch_size_per_sample} embeddings)")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 获取一组embeddings
        start_idx = idx * self.batch_size_per_sample
        end_idx = min(start_idx + self.batch_size_per_sample, len(self.embeddings))
        
        # 提取embeddings: (batch_size_per_sample, 768)
        sample_embs = self.embeddings[start_idx:end_idx]
        
        # 如果不够，用最后一个embedding填充
        if len(sample_embs) < self.batch_size_per_sample:
            padding = sample_embs[-1:] if len(sample_embs) > 0 else torch.zeros(1, 768)
            num_padding = self.batch_size_per_sample - len(sample_embs)
            sample_embs = torch.cat([sample_embs, padding.repeat(num_padding, 1)], dim=0)
        
        return {
            "embeddings": sample_embs[:self.batch_size_per_sample]  # (batch_size_per_sample, 768)
        }


class CodebookTrainer:
    """第一阶段训练器：只训练codebook，不涉及LLM和encoder微调"""
    def __init__(self, codebook_model, accelerator, 
                 codebook_weight=1.0, diversity_weight=0.1, usage_weight=0.3):
        self.codebook_model = codebook_model
        self.accelerator = accelerator
        self.codebook_weight = codebook_weight
        self.diversity_weight = diversity_weight
        self.usage_weight = usage_weight
        self.base_diversity_weight = diversity_weight
        self.collapse_detected = False
        # 用于累积epoch统计
        self.epoch_indices_list = []
        
    def compute_loss(self, batch):
        """计算loss（只包含codebook相关loss）"""
        embeddings = batch["embeddings"]  # (batch_size, batch_size_per_sample, 768)
        device = embeddings.device
        batch_size, batch_size_per_sample, emb_dim = embeddings.shape
        
        # 对embeddings进行Product Quantization
        quantized_embs, indices = self.codebook_model(embeddings)
        # indices: (batch_size, batch_size_per_sample, num_subspaces)
        
        # 诊断：检查embeddings的分散程度（仅在第一个batch打印）
        if not hasattr(self, '_diagnostic_printed'):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                # 检查原始embeddings的分散程度
                flat_embs = embeddings.view(-1, emb_dim)
                emb_std = flat_embs.std(dim=0).mean().item()
                emb_mean_dist = torch.cdist(flat_embs, flat_embs, p=2)
                mask = ~torch.eye(emb_mean_dist.size(0), dtype=torch.bool, device=emb_mean_dist.device)
                emb_mean_dist_value = emb_mean_dist[mask].mean().item() if mask.any() else 0.0
                
                print(f"\n[Initial Diagnostic - PQ]")
                print(f"  Embeddings std: {emb_std:.4f}")
                print(f"  Embeddings mean distance: {emb_mean_dist_value:.4f}")
                print(f"  Number of subspaces: {self.codebook_model.num_subspaces}")
                print(f"  Subspace dimension: {self.codebook_model.subspace_dim}")
            self._diagnostic_printed = True
        
        # Codebook loss: VQ-VAE风格的loss
        # Commitment loss: 让embeddings接近codebook
        commitment_loss = F.mse_loss(quantized_embs.detach(), embeddings)
        # Codebook loss: 让codebook接近embeddings
        codebook_loss = F.mse_loss(quantized_embs, embeddings.detach())
        total_codebook_loss = commitment_loss + codebook_loss
        
        # Diversity loss: 防止codebook塌缩（对每个子空间的codebook分别计算）
        codebook_size = self.codebook_model.codebook_size
        num_subspaces = self.codebook_model.num_subspaces
        
        # 统计所有子空间的codebook使用情况
        flat_indices = indices.view(-1, num_subspaces)  # (batch * seq_len, num_subspaces)
        
        # 计算每个子空间使用的唯一索引数量
        all_unique_counts = []
        all_usage_counts = []
        all_codebooks = []
        
        for i in range(num_subspaces):
            subspace_indices = flat_indices[:, i]  # (batch * seq_len,)
            unique_indices = torch.unique(subspace_indices)
            all_unique_counts.append(len(unique_indices))
            
            # 统计每个codebook entry的使用次数
            usage_counts = torch.zeros(codebook_size, device=indices.device, dtype=torch.long)
            unique_vals, counts = torch.unique(subspace_indices, return_counts=True)
            usage_counts[unique_vals] = counts
            all_usage_counts.append(usage_counts)
            
            all_codebooks.append(self.codebook_model.codebooks[i])
        
        # 统计信息：对于PQ，我们关注每个子空间的使用情况和组合多样性
        # 计算每个子空间的唯一索引数（用于显示）
        all_unique_counts_tensor = torch.tensor(all_unique_counts, device=device, dtype=torch.float)
        avg_unique_per_subspace = all_unique_counts_tensor.mean()  # 平均每个子空间使用的唯一索引数
        
        # 合并所有子空间的使用统计（用于计算总体使用率）
        all_usage_counts_merged = sum(all_usage_counts)  # 合并所有子空间的使用统计
        total_usage = all_usage_counts_merged.sum().float()
        usage_freq = all_usage_counts_merged.float() / (total_usage + 1e-10)
        
        # 计算总体唯一索引数（跨所有子空间，去重后）
        num_unique_total = (all_usage_counts_merged > 0).sum().item()
        
        if avg_unique_per_subspace > 1:
            # 计算所有子空间的codebook entries之间的距离
            all_min_distances = []
            for i in range(num_subspaces):
                sub_unique_indices = torch.unique(flat_indices[:, i])
                if len(sub_unique_indices) > 0:
                    used_codebook = all_codebooks[i][sub_unique_indices]  # (num_unique_sub, subspace_dim)
                    
                    # 计算每个子空间内部的距离
                    if used_codebook.size(0) > 1:
                        distances = torch.cdist(used_codebook, used_codebook, p=2)
                        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
                        if mask.any():
                            min_dist = distances[mask].min()
                            all_min_distances.append(min_dist)
            
            if len(all_min_distances) > 0:
                min_distance = torch.stack(all_min_distances).mean()
            else:
                min_distance = torch.tensor(0.0, device=indices.device)
            
            # 使用tanh平滑的diversity loss
            scale = 5.0
            diversity_loss = -scale * torch.tanh(min_distance / scale)
            
            # Usage loss: 鼓励使用更多的codebook条目
            # 对于PQ，我们不仅要鼓励总体使用率高，还要鼓励不同子空间使用不同的entries
            # 1. 总体使用率
            num_used = (all_usage_counts_merged > 0).sum().float()
            usage_ratio = num_used / codebook_size  # 0-1之间，1表示全部使用
            
            # 2. 子空间多样性：鼓励不同子空间使用不同的entries组合
            # 计算每个子空间的使用率，然后鼓励它们都高
            subspace_usage_ratios = []
            for i in range(num_subspaces):
                sub_usage = (all_usage_counts[i] > 0).sum().float()
                sub_ratio = sub_usage / codebook_size
                subspace_usage_ratios.append(sub_ratio)
            
            # 平均子空间使用率
            avg_subspace_usage = torch.stack(subspace_usage_ratios).mean()
            
            # 3. 组合多样性：鼓励使用不同的子空间组合
            # 计算有多少种不同的子空间索引组合被使用
            # 将每个样本的索引组合转换为唯一标识
            unique_combinations = set()
            flat_indices_cpu = flat_indices.cpu().numpy()
            for idx_tuple in flat_indices_cpu:
                combo = tuple(idx_tuple)
                unique_combinations.add(combo)
            num_unique_combos = len(unique_combinations)
            max_combos = codebook_size ** num_subspaces
            total_samples = batch_size * batch_size_per_sample
            combo_ratio = torch.tensor(num_unique_combos / min(max_combos, total_samples), device=device)
            
            # Usage loss包含三部分：
            # 1. 总体使用率惩罚
            overall_usage_loss = 1.0 - usage_ratio
            # 2. 子空间使用率惩罚（鼓励每个子空间都充分利用）
            subspace_usage_loss = 1.0 - avg_subspace_usage
            # 3. 组合多样性惩罚（鼓励使用不同的组合）
            combo_diversity_loss = 1.0 - combo_ratio
            
            # 加权组合
            usage_ratio_loss = 0.4 * overall_usage_loss + 0.3 * subspace_usage_loss + 0.3 * combo_diversity_loss
            
            non_zero_mask = usage_freq > 1e-10
            if non_zero_mask.sum() > 0:
                entropy = -(usage_freq[non_zero_mask] * torch.log(usage_freq[non_zero_mask] + 1e-10)).sum()
                max_entropy = torch.log(torch.tensor(float(codebook_size), device=device))
                normalized_entropy = entropy / (max_entropy + 1e-10)
                entropy_loss = 1.0 - normalized_entropy
            else:
                entropy_loss = torch.tensor(1.0, device=device)
            
            # 总usage_loss：使用率惩罚（已包含总体、子空间、组合多样性）+ 熵惩罚
            # 使用率惩罚权重更大，因为这是主要目标
            usage_loss = 0.7 * usage_ratio_loss + 0.3 * entropy_loss
            
            # 诊断信息
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1
            
            if self._step_count % 100 == 0:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    # 计算统计信息
                    total_unique_ratio = num_unique_total / codebook_size
                    used_codebook_ratio = (all_usage_counts_merged > 0).sum().item() / codebook_size
                    top10_usage = all_usage_counts_merged.topk(min(10, codebook_size)).values.float() / total_usage if total_usage > 0 else torch.zeros(10)
                    
                    # 计算平均距离（从所有子空间的距离中取平均）
                    if len(all_min_distances) > 0:
                        mean_distance = torch.stack(all_min_distances).mean().item()
                        max_distance = torch.stack(all_min_distances).max().item()
                    else:
                        mean_distance = 0.0
                        max_distance = 0.0
                    
                    entropy = -(usage_freq[non_zero_mask] * torch.log(usage_freq[non_zero_mask] + 1e-10)).sum().item() if non_zero_mask.sum() > 0 else 0.0
                    max_entropy = torch.log(torch.tensor(float(codebook_size), device=device)).item()
                    normalized_entropy = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0.0
                    
                    # 计算每个子空间的使用情况
                    subspace_usage_info = []
                    for i in range(num_subspaces):
                        sub_unique = all_unique_counts[i]
                        sub_ratio = sub_unique / codebook_size
                        subspace_usage_info.append(f"Sub{i}: {sub_unique}/{codebook_size} ({sub_ratio*100:.1f}%)")
                    
                    # 计算组合多样性信息（需要在诊断时重新计算，因为变量作用域）
                    # Unique combinations: 在PQ中，每个embedding被量化成2个索引的组合，比如 (207, 216)
                    # 这个指标统计当前batch中有多少种不同的索引组合被使用
                    unique_combinations_diag = set()
                    flat_indices_diag = indices.view(-1, num_subspaces).cpu().numpy()
                    for idx_tuple in flat_indices_diag:
                        combo = tuple(idx_tuple)
                        unique_combinations_diag.add(combo)
                    num_unique_combos_diag = len(unique_combinations_diag)
                    total_samples_in_batch = batch_size * batch_size_per_sample
                    max_possible_combos = codebook_size ** num_subspaces
                    combo_ratio_diag = num_unique_combos_diag / total_samples_in_batch  # 相对于batch中的样本数
                    combo_coverage = num_unique_combos_diag / min(max_possible_combos, total_samples_in_batch)  # 相对于可能的最大组合数
                    
                    print(f"\n[Step {self._step_count}]")
                    print(f"  Total unique indices (across all subspaces): {num_unique_total} / {codebook_size} ({total_unique_ratio*100:.2f}%)")
                    print(f"  Avg unique per subspace: {avg_unique_per_subspace.item():.1f} / {codebook_size}")
                    print(f"  Per subspace: {', '.join(subspace_usage_info)}")
                    print(f"  Unique combinations: {num_unique_combos_diag} / {total_samples_in_batch} samples ({combo_ratio_diag*100:.2f}% unique)")
                    print(f"    (Each embedding is quantized to 2 indices, e.g., (207,216). This counts how many different 2-tuples are used.)")
                    print(f"    (Max possible combinations: {max_possible_combos:.2e}, Coverage: {combo_coverage*100:.2f}%)")
                    print(f"  Min distance: {min_distance.item():.4f}")
                    print(f"  Mean distance: {mean_distance:.4f}")
                    print(f"  Max distance: {max_distance:.4f}")
                    print(f"  Codebook loss: {total_codebook_loss.item():.6f} (commitment: {commitment_loss.item():.6f}, codebook: {codebook_loss.item():.6f})")
                    print(f"  Diversity loss: {diversity_loss.item():.4f}")
                    print(f"  Usage loss: {usage_loss.item():.4f} (overall: {overall_usage_loss.item():.4f}, subspace: {subspace_usage_loss.item():.4f}, combo: {combo_diversity_loss.item():.4f}, entropy: {entropy_loss.item():.4f})")
                    print(f"  Overall usage ratio: {usage_ratio.item():.4f} ({num_used.item():.0f}/{codebook_size})")
                    print(f"  Avg subspace usage: {avg_subspace_usage.item():.4f}")
                    print(f"  Combination diversity: {combo_ratio.item():.4f}")
                    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} (normalized: {normalized_entropy:.4f})")
                    print(f"  Top-10 usage: {top10_usage.tolist()}")
                    
                    if avg_unique_per_subspace < codebook_size * 0.01:
                        print(f"  ⚠️  WARNING: Codebook collapse! Only {avg_unique_per_subspace.item():.1f} unique indices per subspace on average.")
        elif avg_unique_per_subspace <= 1:
            diversity_loss = torch.tensor(10.0, device=indices.device)
            # 最大惩罚：使用率=0，熵=0
            usage_loss = torch.tensor(1.0, device=indices.device)
            
            if not self.collapse_detected:
                self.collapse_detected = True
                self.diversity_weight = self.base_diversity_weight * 5.0
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    # 检查embeddings的分散程度
                    flat_embs = embeddings.view(-1, emb_dim)
                    emb_std = flat_embs.std(dim=0).mean().item()
                    emb_mean_dist = torch.cdist(flat_embs, flat_embs, p=2)
                    mask = ~torch.eye(emb_mean_dist.size(0), dtype=torch.bool, device=emb_mean_dist.device)
                    emb_mean_dist_value = emb_mean_dist[mask].mean().item() if mask.any() else 0.0
                    
                    print(f"\n⚠️  Codebook collapse detected! Only 1 entry used.")
                    print(f"  Embeddings std: {emb_std:.4f}")
                    print(f"  Embeddings mean distance: {emb_mean_dist_value:.4f}")
                    print(f"  Increasing diversity_weight to {self.diversity_weight}")
                    print(f"  ⚠️  If embeddings are too similar, codebook may collapse!")
        else:
            diversity_loss = torch.tensor(10.0, device=indices.device)
            # 最大惩罚：使用率=0，熵=0
            usage_loss = torch.tensor(1.0, device=indices.device)
        
        # 总loss（只包含codebook相关loss）
        # 对于PQ，我们需要平衡：
        # 1. Codebook loss: 量化误差（主要目标，权重1.0）
        # 2. Diversity loss: 防止codebook塌缩（负数，权重较小0.1-0.2）
        # 3. Usage loss: 鼓励使用更多entries和组合（权重较大5.0-10.0，因为PQ的优势在于组合）
        total_loss = (self.codebook_weight * total_codebook_loss + 
                     self.diversity_weight * diversity_loss +
                     self.usage_weight * usage_loss)
        
        # 累积indices用于epoch统计（只在主进程累积）
        if hasattr(self, '_collecting_epoch_stats') and self._collecting_epoch_stats:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                # 将indices移到CPU并保存
                indices_cpu = indices.detach().cpu().numpy()
                self.epoch_indices_list.append(indices_cpu)
        
        return total_loss, total_codebook_loss, diversity_loss, usage_loss
    
    def start_epoch_stats(self):
        """开始收集epoch统计信息"""
        self._collecting_epoch_stats = True
        self.epoch_indices_list = []
    
    def end_epoch_stats(self):
        """结束收集并计算epoch统计信息"""
        if not hasattr(self, '_collecting_epoch_stats') or not self._collecting_epoch_stats:
            return None
        
        if len(self.epoch_indices_list) == 0:
            self._collecting_epoch_stats = False
            return None
        
        # 合并所有batch的indices
        # epoch_indices_list: list of (batch_size, batch_size_per_sample, num_subspaces)
        all_indices = np.concatenate(self.epoch_indices_list, axis=0)  # (total_samples, batch_size_per_sample, num_subspaces)
        flat_indices = all_indices.reshape(-1, self.codebook_model.num_subspaces)  # (total_samples * batch_size_per_sample, num_subspaces)
        
        codebook_size = self.codebook_model.codebook_size
        num_subspaces = self.codebook_model.num_subspaces
        
        # 统计每个子空间的使用情况
        subspace_stats = []
        all_used_indices_set = set()
        
        for subspace_idx in range(num_subspaces):
            subspace_indices = flat_indices[:, subspace_idx]  # (total_samples * batch_size_per_sample,)
            unique_subspace_indices = np.unique(subspace_indices)
            num_unique_subspace = len(unique_subspace_indices)
            usage_ratio_subspace = num_unique_subspace / codebook_size
            
            # 收集这个子空间使用的所有词条索引（用于统计总数）
            all_used_indices_set.update(unique_subspace_indices.tolist())
            
            subspace_stats.append({
                'subspace_idx': subspace_idx,
                'num_unique': num_unique_subspace,
                'usage_ratio': usage_ratio_subspace
            })
        
        # 统计所有子空间一共使用了多少个唯一的词条（去重后）
        total_unique_entries = len(all_used_indices_set)
        # 注意：两个subspace共用一个codebook，所以total_possible_entries = codebook_size
        total_possible_entries = codebook_size
        total_usage_ratio = total_unique_entries / total_possible_entries if total_possible_entries > 0 else 0.0
        
        # 统计唯一组合数
        unique_combinations = set()
        for idx_tuple in flat_indices:
            combo = tuple(idx_tuple)
            unique_combinations.add(combo)
        num_unique_combos = len(unique_combinations)
        max_possible_combos = min(codebook_size ** num_subspaces, len(flat_indices))
        combo_ratio = num_unique_combos / max_possible_combos if max_possible_combos > 0 else 0.0
        
        # 清理
        self.epoch_indices_list = []
        self._collecting_epoch_stats = False
        
        return {
            'subspace_stats': subspace_stats,
            'total_unique_entries': total_unique_entries,
            'total_possible_entries': total_possible_entries,
            'total_usage_ratio': total_usage_ratio,
            'num_unique_combos': num_unique_combos,
            'max_possible_combos': max_possible_combos,
            'combo_ratio': combo_ratio,
            'total_samples': len(flat_indices)
        }


def train(args):
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print(f"Stage 1: Training codebook only (without LLM)")
        print(f"Codebook size: {args.codebook_size}")
        print(f"Codebook dim: 768 (same as encoder embeddings)")
        print(f"Batch size per sample: {args.batch_size_per_sample}")
    
    # 注意：encoder只在初始化时使用（如果embeddings_path未提供），不进行微调
    
    # 创建codebook模型（Product Quantization版本：2个子空间独立量化）
    codebook_model = CodebookModel(
        codebook_size=args.codebook_size,
        emb_dim=768,  # 与encoder embeddings维度一致
        num_subspaces=2  # 分成2个子空间，每个384维
    )
    
    # Balanced K-means初始化
    if args.use_balanced_kmeans_init:
        init_codebook_path = os.path.join(
            args.output_dir, 
            f"codebook_init_size{args.codebook_size}_dim768_pq2_sample{args.init_sample_size if args.init_sample_size else 'all'}.pt"
        )
        
        if os.path.exists(init_codebook_path):
            if accelerator.is_local_main_process:
                print(f"Loading saved PQ initialization from {init_codebook_path}...")
            init_data = torch.load(init_codebook_path, map_location='cpu')
            
            # 检查是否是PQ格式（dict with 'codebooks' key）或旧格式（单个tensor）
            if isinstance(init_data, dict) and 'codebooks' in init_data:
                # PQ格式：加载每个子空间的codebook
                init_codebooks = init_data['codebooks']
                if len(init_codebooks) == codebook_model.num_subspaces:
                    for i, init_cb in enumerate(init_codebooks):
                        if init_cb.shape == (args.codebook_size, codebook_model.subspace_dim):
                            codebook_model.codebooks[i].data = init_cb
                        else:
                            raise ValueError(f"Subspace {i} shape mismatch! Expected ({args.codebook_size}, {codebook_model.subspace_dim}), got {init_cb.shape}")
                    if accelerator.is_local_main_process:
                        print("✅ Loaded saved PQ initialization!")
                else:
                    raise ValueError(f"Number of subspaces mismatch! Expected {codebook_model.num_subspaces}, got {len(init_codebooks)}")
            elif isinstance(init_data, torch.Tensor):
                # 旧格式：单个codebook，需要拆分成子空间
                if init_data.shape == (args.codebook_size, 768):
                    if accelerator.is_local_main_process:
                        print("Converting old format to PQ format...")
                    # 将768维codebook拆分成2个子空间
                    for i in range(codebook_model.num_subspaces):
                        start_idx = i * codebook_model.subspace_dim
                        end_idx = (i + 1) * codebook_model.subspace_dim
                        codebook_model.codebooks[i].data = init_data[:, start_idx:end_idx]
                    if accelerator.is_local_main_process:
                        print("✅ Converted and loaded initialization!")
                else:
                    raise ValueError(f"Shape mismatch! Expected ({args.codebook_size}, 768), got {init_data.shape}")
            else:
                raise ValueError(f"Unknown initialization format: {type(init_data)}")
            
            accelerator.wait_for_everyone()
        else:
            # 需要初始化
            if accelerator.is_local_main_process:
                print("Initializing codebook with Balanced K-means...")
            
            temp_codebook_path = os.path.join(args.output_dir, "temp_codebook_init.pt")
            
            if accelerator.is_local_main_process:
                # 优先使用预计算的embeddings文件（如果提供）
                if args.embeddings_path and os.path.exists(args.embeddings_path):
                    print(f"Loading pre-computed embeddings from {args.embeddings_path}...")
                    init_embeddings = torch.load(args.embeddings_path, map_location='cpu')
                    
                    # 处理不同的数据格式
                    if isinstance(init_embeddings, torch.Tensor):
                        if init_embeddings.dim() == 2:
                            pass  # 已经是 (N, 768) 格式
                        else:
                            raise ValueError(f"Expected 2D tensor (N, 768), got shape {init_embeddings.shape}")
                    elif isinstance(init_embeddings, list):
                        init_embeddings = torch.stack(init_embeddings)
                    else:
                        raise ValueError(f"Unknown embeddings format: {type(init_embeddings)}")
                    
                    print(f"Loaded {len(init_embeddings)} pre-computed embeddings")
                    
                    if args.init_sample_size and len(init_embeddings) > args.init_sample_size:
                        print(f"Sampling {args.init_sample_size} embeddings...")
                        sample_indices = torch.randperm(len(init_embeddings))[:args.init_sample_size]
                        init_embeddings = init_embeddings[sample_indices]
                else:
                    # 如果没有预计算文件，从JSONL实时编码
                    print("No pre-computed embeddings provided, encoding from JSONL...")
                    encoder_model.eval()
                    
                    # 从JSONL文件读取数据并编码
                    print("Reading data and encoding embeddings for initialization...")
                    init_embeddings_list = []
                    with open(args.data_path, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            if args.init_sample_size and len(init_embeddings_list) >= args.init_sample_size:
                                break
                            item = json.loads(line.strip())
                            user_content = item.get("User-Generated Content", [])
                            for content_item in user_content:
                                if args.init_sample_size and len(init_embeddings_list) >= args.init_sample_size:
                                    break
                                prompt_text = content_item.get("prompt", "").strip()
                                comment_text = content_item.get("comment", "").strip()
                                if prompt_text and comment_text:
                                    text = f"Question: {prompt_text} Answer: {comment_text}"
                                    with torch.no_grad():
                                        encoder_device = next(encoder_model.parameters()).device
                                        inputs = encoder_tokenizer(
                                            text,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt"
                                        ).to(encoder_device)
                                        outputs = encoder_model(**inputs)
                                        embeddings = outputs.last_hidden_state
                                        attention_mask = inputs['attention_mask']
                                        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                                        summed = torch.sum(masked_embeddings, dim=1)
                                        summed_mask = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
                                        mean_pooled = summed / summed_mask
                                        init_embeddings_list.append(mean_pooled.squeeze(0).cpu())
                    
                    init_embeddings = torch.stack(init_embeddings_list)
                    print(f"Encoded {len(init_embeddings)} embeddings for initialization")
                    
                    if args.init_sample_size and len(init_embeddings) > args.init_sample_size:
                        print(f"Sampling {args.init_sample_size} embeddings...")
                        sample_indices = torch.randperm(len(init_embeddings))[:args.init_sample_size]
                        init_embeddings = init_embeddings[sample_indices]
                
                init_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                print(f"Using embeddings directly (768 dim) for PQ initialization...")
                
                # 对每个子空间独立进行K-means初始化
                # 为了确保不同子空间学习到不同的表示，我们对每个子空间使用不同的随机采样
                print(f"Initializing {codebook_model.num_subspaces} subspaces independently...")
                initialized_codebooks = []
                
                for i in range(codebook_model.num_subspaces):
                    start_idx = i * codebook_model.subspace_dim
                    end_idx = (i + 1) * codebook_model.subspace_dim
                    subspace_embs = init_embeddings[:, start_idx:end_idx]  # (N, subspace_dim)
                    
                    # 为了增加多样性，对每个子空间使用不同的随机采样（如果样本数足够）
                    if len(subspace_embs) > args.init_sample_size and args.init_sample_size:
                        # 每个子空间使用不同的随机种子进行采样
                        sample_indices = torch.randperm(len(subspace_embs), generator=torch.Generator().manual_seed(i))[:args.init_sample_size]
                        subspace_embs = subspace_embs[sample_indices]
                    
                    print(f"  Initializing subspace {i+1}/{codebook_model.num_subspaces} (dim {start_idx}:{end_idx}) with {len(subspace_embs)} samples...")
                    initialized_subspace = balanced_kmeans_init(
                        embeddings=subspace_embs,
                        num_clusters=args.codebook_size,
                        max_iters=args.kmeans_max_iters,
                        device=init_device
                    )
                    initialized_codebooks.append(initialized_subspace.cpu())
                    codebook_model.codebooks[i].data = initialized_subspace.cpu()
                
                os.makedirs(args.output_dir, exist_ok=True)
                # 保存为PQ格式
                torch.save({
                    'codebooks': initialized_codebooks,
                    'num_subspaces': codebook_model.num_subspaces,
                    'subspace_dim': codebook_model.subspace_dim
                }, init_codebook_path)
                print(f"✅ Saved PQ initialization to {init_codebook_path}")
                
                import shutil
                temp_file = temp_codebook_path + ".tmp"
                # 保存PQ格式
                torch.save({
                    'codebooks': [cb.data.cpu() for cb in codebook_model.codebooks],
                    'num_subspaces': codebook_model.num_subspaces,
                    'subspace_dim': codebook_model.subspace_dim
                }, temp_file)
                shutil.move(temp_file, temp_codebook_path)
                print(f"Saved temporary PQ file for other processes")
            else:
                # 其他进程等待
                import time
                max_wait_time = 7200
                wait_interval = 10
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    if os.path.exists(temp_codebook_path):
                        try:
                            file_size = os.path.getsize(temp_codebook_path)
                            # PQ格式：2个子空间，每个codebook_size * subspace_dim
                            expected_size = codebook_model.num_subspaces * args.codebook_size * codebook_model.subspace_dim * 4  # float32
                            if file_size >= expected_size * 0.9:
                                init_data = torch.load(temp_codebook_path, map_location='cpu')
                                # 检查格式
                                if isinstance(init_data, dict) and 'codebooks' in init_data:
                                    init_codebooks = init_data['codebooks']
                                    if len(init_codebooks) == codebook_model.num_subspaces:
                                        for i, init_cb in enumerate(init_codebooks):
                                            if init_cb.shape == (args.codebook_size, codebook_model.subspace_dim):
                                                codebook_model.codebooks[i].data = init_cb
                                            else:
                                                print(f"Shape mismatch for subspace {i}")
                                                break
                                        else:
                                            print("PQ codebook loaded!")
                                            break
                                elif isinstance(init_data, torch.Tensor):
                                    # 旧格式，转换
                                    if init_data.shape == (args.codebook_size, 768):
                                        for i in range(codebook_model.num_subspaces):
                                            start_idx = i * codebook_model.subspace_dim
                                            end_idx = (i + 1) * codebook_model.subspace_dim
                                            codebook_model.codebooks[i].data = init_data[:, start_idx:end_idx]
                                        print("Codebook converted and loaded!")
                                        break
                        except Exception as e:
                            print(f"File not ready: {e}")
                    
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                    if elapsed_time % 60 == 0:
                        print(f"Waiting... ({elapsed_time//60}min)")
                
                if not os.path.exists(temp_codebook_path):
                    raise RuntimeError(f"Timeout after {max_wait_time//60}min")
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_local_main_process and os.path.exists(temp_codebook_path):
                try:
                    os.remove(temp_codebook_path)
                except:
                    pass
    else:
        if accelerator.is_local_main_process:
            print("Using random initialization")
    
    # 创建数据集（使用预计算的embeddings）
    dataset = CodebookDataset(
        embeddings_path=args.embeddings_path,
        batch_size_per_sample=args.batch_size_per_sample,
        device="cpu"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建训练器
    trainer = CodebookTrainer(
        codebook_model=codebook_model,
        accelerator=accelerator,
        codebook_weight=args.codebook_weight,
        diversity_weight=args.diversity_weight,
        usage_weight=args.usage_weight
    )
    
    # 创建优化器（只优化codebook）
    optimizer = torch.optim.AdamW(
        codebook_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 使用accelerate准备
    codebook_model, optimizer, dataloader = accelerator.prepare(
        codebook_model, optimizer, dataloader
    )
    
    # 训练循环
    codebook_model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 开始收集epoch统计
        if accelerator.is_local_main_process:
            trainer.start_epoch_stats()
        
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            total_loss, codebook_loss, diversity_loss, usage_loss = trainer.compute_loss(batch)
            
            accelerator.backward(total_loss)
            
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(codebook_model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            global_step += 1
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "codebook": f"{codebook_loss.item():.4f}",
                    "diversity": f"{diversity_loss.item():.4f}",
                    "usage": f"{usage_loss.item():.4f}"
                })
            
            # 保存checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_local_main_process:
                    save_path = os.path.join(args.output_dir, f"stage1_checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_model = accelerator.unwrap_model(codebook_model)
                    checkpoint_dict = {
                        "codebooks": [cb.data.cpu() for cb in unwrapped_model.codebooks],
                        "num_subspaces": unwrapped_model.num_subspaces,
                        "subspace_dim": unwrapped_model.subspace_dim,
                        "step": global_step,
                        "codebook_size": args.codebook_size,
                        "emb_dim": 768
                    }
                    
                    torch.save(checkpoint_dict, os.path.join(save_path, "codebook_model.pt"))
                    print(f"Saved checkpoint to {save_path}")
        
        # 计算并打印epoch统计
        if accelerator.is_local_main_process:
            epoch_stats = trainer.end_epoch_stats()
            if epoch_stats is not None:
                unwrapped_model = accelerator.unwrap_model(codebook_model)
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{args.num_epochs} - Codebook Usage Statistics")
                print(f"{'='*60}")
                print(f"Total samples processed: {epoch_stats['total_samples']}")
                print(f"\nPer Subspace Usage:")
                for sub_stat in epoch_stats['subspace_stats']:
                    print(f"  Subspace {sub_stat['subspace_idx']}: {sub_stat['num_unique']} / {unwrapped_model.codebook_size} unique entries used ({sub_stat['usage_ratio']*100:.2f}%)")
                print(f"\nOverall Usage (across all subspaces, deduplicated):")
                print(f"  Total unique codes used: {epoch_stats['total_unique_entries']} / {epoch_stats['total_possible_entries']} ({epoch_stats['total_usage_ratio']*100:.2f}%)")
                print(f"\nCombination Diversity:")
                print(f"  Unique combinations: {epoch_stats['num_unique_combos']} / {epoch_stats['max_possible_combos']} ({epoch_stats['combo_ratio']*100:.2f}%)")
                print(f"{'='*60}\n")
    
    # 最终保存
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "stage1_final")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(codebook_model)
        final_dict = {
            "codebooks": [cb.data.cpu() for cb in unwrapped_model.codebooks],
            "num_subspaces": unwrapped_model.num_subspaces,
            "subspace_dim": unwrapped_model.subspace_dim,
            "codebook_size": args.codebook_size,
            "emb_dim": 768
        }
        
        torch.save(final_dict, os.path.join(final_path, "codebook_model.pt"))
        print(f"Stage 1 training completed! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Train codebook only (without LLM and encoder fine-tuning)")
    
    parser.add_argument("--embeddings_path", type=str, required=True,
                       help="Path to pre-computed embeddings .pt file")
    parser.add_argument("--encoder_path", type=str, default=None,
                       help="Path to encoder model (Contriever, optional, only for initialization if embeddings_path not provided)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to JSONL data file (optional, only for initialization if embeddings_path not provided)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    parser.add_argument("--codebook_size", type=int, default=10000,
                       help="Codebook size")
    parser.add_argument("--batch_size_per_sample", type=int, default=8,
                       help="Number of embeddings per sample (corresponds to his_len)")
    
    parser.add_argument("--codebook_weight", type=float, default=1.0,
                       help="Weight for codebook loss")
    parser.add_argument("--diversity_weight", type=float, default=0.1,
                       help="Weight for diversity loss")
    parser.add_argument("--usage_weight", type=float, default=0.3,
                       help="Weight for usage loss")
    
    parser.add_argument("--use_balanced_kmeans_init", action="store_true",
                       help="Use Balanced K-means initialization")
    parser.add_argument("--init_sample_size", type=int, default=None,
                       help="Number of embeddings to sample for initialization")
    parser.add_argument("--kmeans_max_iters", type=int, default=100,
                       help="Max iterations for Balanced K-means")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (number of samples)")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    train(args)

