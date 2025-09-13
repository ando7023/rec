from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb
import numpy as np
import torch
import torch.nn.functional as F


class RoPE(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache()
    
    def _build_cache(self):
        seq_len = self.max_seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # 将频率扩展到完整维度
        emb = torch.cat([freqs, freqs], dim=-1)
        # 缓存cos和sin值
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def rotate_half(self, x):
        """将输入的后半部分旋转到前半部分"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """
        应用RoPE到query和key
        Args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_heads, seq_len, head_dim]
            seq_len: 实际序列长度（如果不同于q/k的长度）
        """
        if seq_len is None:
            seq_len = q.shape[2]
        
        # 确保维度是偶数
        assert q.shape[-1] == self.dim, f"Feature dimension {q.shape[-1]} doesn't match RoPE dimension {self.dim}"
        
        # 获取对应长度的cos和sin
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        
        # 应用旋转
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

class HSTUBlock(torch.nn.Module):
    """
    实现了 HSTU (Hierarchical Sequential Transduction Unit) 的一个 Block，带有RoPE位置编码。
    【版本】: 支持 QK 和 UV 维度解耦，并与官方核心算法对齐，添加RoPE支持。
    """

    def __init__(self, hidden_units, num_heads, dropout_rate,
                 attn_head_dim, linear_head_dim, use_rope=True, max_seq_len=512,
                 rope_base=10000, rab_module=None):
        super(HSTUBlock, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attn_head_dim = attn_head_dim
        self.linear_head_dim = linear_head_dim
        self.use_rope = use_rope
        self.rab_module = rab_module

        # 输入的 LayerNorm
        self.input_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        # 定义 uvqk_linear
        uvqk_output_dim = (linear_head_dim * num_heads * 2) + \
                          (attn_head_dim * num_heads * 2)
        self.uvqk_linear = torch.nn.Linear(hidden_units, uvqk_output_dim)

        # 注意力输出的 LayerNorm
        attn_output_dim = linear_head_dim * num_heads
        self.attn_output_norm = torch.nn.LayerNorm(attn_output_dim, eps=1e-8)

        # 注意力权重 dropout
        self.attn_dropout = torch.nn.Dropout(p=dropout_rate)

        # 输出线性层
        self.o_linear = torch.nn.Linear(attn_output_dim, hidden_units)

        # 输出 dropout
        self.resid_dropout = torch.nn.Dropout(p=dropout_rate)
        
        # 初始化RoPE（如果启用）
        if self.use_rope:
            # 确保attn_head_dim是偶数（RoPE要求）
            assert attn_head_dim % 2 == 0, f"attn_head_dim must be even for RoPE, got {attn_head_dim}"
            self.rope = RoPE(dim=attn_head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x, attn_mask=None, timestamps=None):
        batch_size, seq_len, _ = x.shape

        residual = x

        # 1. 输入归一化和统一投射
        x_norm = self.input_norm(x)
        uvqk = self.uvqk_linear(x_norm)
        uvqk_activated = F.silu(uvqk)

        # 使用 torch.split 按精确维度分割 u, v, q, k
        u_dim = v_dim = self.linear_head_dim * self.num_heads
        q_dim = k_dim = self.attn_head_dim * self.num_heads

        u, v, q, k = torch.split(uvqk_activated, [u_dim, v_dim, q_dim, k_dim], dim=-1)

        # 2. 空间聚合 (SiLU Attention)
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_head_dim).transpose(1, 2)

        # 应用RoPE到q和k（如果启用）
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)

        # 计算 attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 添加相对注意力偏差（如果提供）
        if self.rab_module is not None and timestamps is not None:
            scores += self.rab_module(timestamps).unsqueeze(1)

        attn_weights = F.silu(scores) / seq_len

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).logical_not(), 0)

        attn_weights = self.attn_dropout(attn_weights)

        # 计算 attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape 回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 3. 输出转换 (Pointwise Transformation)
        attn_output_norm = self.attn_output_norm(attn_output)

        # 门控操作
        gated_output = attn_output_norm * u

        # 输出线性层和dropout
        output = self.resid_dropout(self.o_linear(gated_output))

        # 4. 残差连接
        output += residual

        return output

# 将这个新模块添加到你的 model.py 文件中
class RelativeAttentionBias(torch.nn.Module):
    def __init__(self, max_seq_len, num_time_buckets=128):
        super().__init__()
        self.max_seq_len = max_seq_len
        # 相对位置偏差的 embedding
        self.pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )
        # 相对时间偏差的 embedding
        self.ts_w = torch.nn.Parameter(
            torch.empty(num_time_buckets + 1).normal_(mean=0, std=0.02) # +1 用于处理padding或0值
        )
        self.num_time_buckets = num_time_buckets

    def _bucketize_time_diff(self, time_diff):
        """将时间差（秒）转换为对数分桶的ID"""
        # 使用对数分桶，对近期的时间差更敏感
        # 这个函数可以根据你的数据分布进行定制
        # clamp(min=1) 避免 log(0)
        return (torch.log(torch.abs(time_diff).clamp(min=1)) / 0.301).long()

    def forward(self, timestamps):
        """
        Args:
            timestamps: (B, N) 的时间戳张量
        Returns:
            (B, N, N) 的偏差矩阵
        """
        batch_size, seq_len = timestamps.shape

        # --- 计算相对位置偏差 ---
        # 这是一个高效计算 relative position bias 矩阵的技巧
        q_pos = torch.arange(seq_len, device=timestamps.device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=timestamps.device).unsqueeze(0)
        relative_pos = k_pos - q_pos
        # 将相对位置 [-N+1, N-1] 映射到 [0, 2N-2] 作为 embedding 索引
        relative_pos_idx = relative_pos + self.max_seq_len - 1
        rel_pos_bias = self.pos_w[relative_pos_idx].expand(batch_size, -1, -1)

        # --- 计算相对时间偏差 ---
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1) # (B, N, N)
        bucketed_time_diff = self._bucketize_time_diff(time_diff)
        # 限制桶ID的最大值
        bucketed_time_diff = torch.clamp(bucketed_time_diff, min=0, max=self.num_time_buckets)

        # 查找时间偏差 embedding
        rel_ts_bias = self.ts_w[bucketed_time_diff]

        return rel_pos_bias + rel_ts_bias



class BaselineModel(torch.nn.Module):

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # self.norm_first = args.norm_first # HSTU 有固定的 Norm 位置, 这个参数不再需要
        self.maxlen = args.maxlen
        self.temp = args.temp

        # 【新增】: 存储ID Dropout比率
        self.id_dropout_rate = args.id_dropout_rate

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        #self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        rab_module = RelativeAttentionBias(max_seq_len=args.maxlen + 1)  # +1 匹配你的代码

        # 【改动 1】: 移除旧的 Transformer Block 模块
        # self.attention_layernorms = torch.nn.ModuleList()
        # self.attention_layers = torch.nn.ModuleList()
        # self.forward_layernorms = torch.nn.ModuleList()
        # self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
                args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
                + len(self.ITEM_CONTINUAL_FEAT)
                + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        time_dim = args.hidden_units * len(self.TIME_FEAT)
        if time_dim > 0:
            self.user_time_dnn = torch.nn.Linear(time_dim, args.hidden_units)
            self.item_time_dnn = torch.nn.Linear(time_dim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 【改动 2】: 使用新的 HSTUBlock 模块
        # self.hstu_blocks = torch.nn.ModuleList(
        #     [HSTUBlock(args.hidden_units, args.num_heads, args.dropout_rate) for _ in range(args.num_blocks)]
        # )
        attn_head_dim = args.attn_head_dim  # From args, default 48
        linear_head_dim = args.linear_head_dim  # From args, default 24
        self.hstu_blocks = torch.nn.ModuleList(
            [HSTUBlock(
                hidden_units=args.hidden_units,
                num_heads=args.num_heads,
                dropout_rate=args.dropout_rate,
                attn_head_dim=attn_head_dim,
                linear_head_dim=linear_head_dim,
                use_rope=True,  # 启用RoPE
                max_seq_len=args.maxlen + 1,
                rope_base=10000,
                rab_module=rab_module
            ) for _ in range(args.num_blocks)]
        )

        all_sparse_feats = {**self.USER_SPARSE_FEAT, **self.ITEM_SPARSE_FEAT,
                            **self.USER_ARRAY_FEAT, **self.ITEM_ARRAY_FEAT,
                            **{k: feat_statistics[k] for k in self.TIME_FEAT}}

        for k, num_embeddings in all_sparse_feats.items():
            self.sparse_emb[k] = torch.nn.Embedding(num_embeddings + 1, args.hidden_units, padding_idx=0)

        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)


    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        # ADDED: Define time features
        self.TIME_FEAT = feat_types['time_sparse']

        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        seq = seq.to(self.dev)

        # MODIFIED: Initialize lists for time features
        item_feat_list = []
        user_feat_list = []
        item_time_feat_list = []
        user_time_feat_list = []

        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list.append(item_embedding)
            user_feat_list.append(user_embedding)
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list.append(item_embedding)

        # Process all feature types
        feature_groups = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list, item_time_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list, None),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list, None),
        ]
        if include_user:
            feature_groups.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list, user_time_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list, None),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list, None),
                (self.TIME_FEAT, 'time_sparse', item_feat_list, item_time_feat_list)  # Time feats can apply to items
            ])
            if include_user:
                # Also associate time features with the user token
                feature_groups.append((self.TIME_FEAT, 'time_sparse_user', user_feat_list, user_time_feat_list))

        for feat_dict, feat_type, main_list, time_list in feature_groups:
            if not feat_dict: continue

            # Special handling for shared time features
            current_feat_keys = feat_dict.keys() if isinstance(feat_dict, dict) else feat_dict

            for k in current_feat_keys:
                if k not in feature_tensors: continue
                tensor_feature = feature_tensors[k].to(self.dev)

                if 'sparse' in feat_type:
                    # MODIFIED: Route to main list or time list
                    if k in self.TIME_FEAT and time_list is not None:
                        time_list.append(self.sparse_emb[k](tensor_feature))
                    elif k not in self.TIME_FEAT:
                        main_list.append(self.sparse_emb[k](tensor_feature))
                elif 'array' in feat_type:
                    emb = self.sparse_emb[k](tensor_feature)
                    main_list.append(emb.sum(2))
                elif 'continual' in feat_type:
                    main_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev)
                item_feat_list.append(self.emb_transform[k](tensor_feature))

        # MODIFIED: Dual DNN Path Fusion
        all_item_emb_main = torch.cat(item_feat_list, dim=2)
        all_item_emb_main = torch.nn.GELU()(self.itemdnn(all_item_emb_main))

        if item_time_feat_list:
            all_item_emb_time = torch.cat(item_time_feat_list, dim=2)
            all_item_emb_time = torch.nn.GELU()(self.item_time_dnn(all_item_emb_time))
            all_item_emb = all_item_emb_main + all_item_emb_time
        else:
            all_item_emb = all_item_emb_main

        if include_user:
            all_user_emb_main = torch.cat(user_feat_list, dim=2)
            all_user_emb_main = torch.nn.GELU()(self.userdnn(all_user_emb_main))

            if user_time_feat_list:
                all_user_emb_time = torch.cat(user_time_feat_list, dim=2)
                all_user_emb_time = torch.nn.GELU()(self.user_time_dnn(all_user_emb_time))
                all_user_emb = all_user_emb_main + all_user_emb_time
            else:
                all_user_emb = all_user_emb_main

            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    # ... (log2feats and other methods remain the same) ...
    def log2feats(self, log_seqs, mask, seq_feature, timestamps):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征tensor字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """

        # 【新增】: ID Dropout 逻辑
        if self.training and self.id_dropout_rate > 0:
            # 1. 创建一个和log_seqs形状相同的随机概率张量
            probs = torch.rand(log_seqs.shape, device=self.dev)

            # 2. 确定需要被dropout的位置 (概率小于阈值)
            dropout_mask = probs < self.id_dropout_rate

            # 3. 确保不对padding tokens (ID=0) 进行dropout
            non_padding_mask = (log_seqs != 0)

            # 4. 结合两个mask，只对非padding的、且被选中的token进行dropout
            final_mask = dropout_mask & non_padding_mask

            # 5. 将被选中的ID替换为padding ID (0)
            # 使用 in-place 操作 `masked_fill_` 提高效率
            log_seqs = log_seqs.masked_fill(final_mask, 0)

        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        # poss *= log_seqs != 0
        # seqs += self.pos_emb(poss)

        seqs = self.emb_dropout(seqs)



        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        #timestamps = seq_feature['timestamp'].to(self.dev)  # 确保 key 和 shape (B, N) 正确

        # 【改动 3】: 简化循环，直接调用 HSTUBlock
        for block in self.hstu_blocks:
            seqs = block(seqs, attn_mask=attention_mask, timestamps=timestamps)

        log_feats = self.last_layernorm(seqs)

        return log_feats


    def compute_infnce_loss(self, seq_embs_raw, pos_embs_raw, neg_embs_raw, loss_mask=None):
        """组合多个改进：Margin + Hard Negative Mining"""

        # 筛选有效样本
        batch_size, maxlen, hidden_size = seq_embs_raw.shape

        if loss_mask is not None:
            if not loss_mask.dtype == torch.bool:
                loss_mask = loss_mask.bool()
            seq_embs = seq_embs_raw[loss_mask]
            pos_embs = pos_embs_raw[loss_mask]
            neg_embs = neg_embs_raw[loss_mask]
        else:
            seq_embs = seq_embs_raw.reshape(-1, hidden_size)
            pos_embs = pos_embs_raw.reshape(-1, hidden_size)
            neg_embs = neg_embs_raw.reshape(-1, hidden_size)

        if seq_embs.numel() == 0:
            return torch.tensor(0.0, device=seq_embs_raw.device, requires_grad=True)

        num_valid_samples = seq_embs.size(0)

        # L2归一化
        seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
        pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
        neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

        # 计算相似度
        pos_logits = torch.sum(seq_embs * pos_embs, dim=-1, keepdim=True)
        neg_logits = torch.matmul(seq_embs, neg_embs.transpose(0, 1))

        # 方案1：使用所有负样本的InfoNCE
        all_logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros(num_valid_samples, device=all_logits.device, dtype=torch.long)
        infonce_loss = F.cross_entropy(all_logits / self.temp, labels)

        # 方案2：Hard Negative的额外loss
        k = min(10, neg_logits.size(1))
        hard_neg_logits, _ = torch.topk(neg_logits, k=k, dim=1, largest=True)
        hard_logits = torch.cat([pos_logits, hard_neg_logits], dim=-1)
        hard_loss = F.cross_entropy(hard_logits / (self.temp * 0.5), labels[:, None].expand(-1, 1).squeeze())

        # 方案3：Margin loss
        margin = 0.25
        hardest_neg = neg_logits.max(dim=1, keepdim=True)[0]
        margin_loss = torch.relu(margin - (pos_logits - hardest_neg)).mean()

        # 组合三个loss
        total_loss = infonce_loss + 0.3 * hard_loss + 0.2 * margin_loss

        return total_loss

    def forward(
            self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
            seq_feature, pos_feature, neg_feature, timestamps  # 增加 timestamps 参数
    ):
        log_feats = self.log2feats(user_item, mask, seq_feature, timestamps)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        return log_feats, pos_embs, neg_embs

    def predict(self, log_seqs, seq_feature, mask, timestamps):
        log_feats = self.log2feats(log_seqs, mask, seq_feature, timestamps)
        final_feat = log_feats[:, -1, :]
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        # ... (This method remains the same but will now implicitly handle time features if they are in feat_dict) ...
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            current_item_ids = item_ids[start_idx:end_idx]

            # 准备特征tensor字典
            batch_feat_tensors = {}

            # 获取所有特征类型
            all_feat_keys = set()
            for feat_type in [self.USER_SPARSE_FEAT, self.ITEM_SPARSE_FEAT, self.USER_ARRAY_FEAT,
                              self.ITEM_ARRAY_FEAT, self.ITEM_EMB_FEAT, self.USER_CONTINUAL_FEAT,
                              self.ITEM_CONTINUAL_FEAT, self.TIME_FEAT]:  # Add time feats here
                if isinstance(feat_type, dict):
                    all_feat_keys.update(feat_type.keys())
                elif isinstance(feat_type, list):
                    all_feat_keys.update(feat_type)

            # 为每个特征准备tensor
            for k in all_feat_keys:
                if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
                    # Array类型特征
                    max_array_len = 0
                    for item_id_re_id in current_item_ids:
                        if k in feat_dict[item_id_re_id]:
                            max_array_len = max(max_array_len, len(feat_dict[item_id_re_id][k]))

                    if max_array_len == 0:
                        max_array_len = 1

                    batch_data = np.zeros((1, len(current_item_ids), max_array_len), dtype=np.int64)
                    for j, item_id_re_id in enumerate(current_item_ids):
                        if k in feat_dict[item_id_re_id]:
                            item_data = feat_dict[item_id_re_id][k]
                            actual_len = min(len(item_data), max_array_len)
                            batch_data[0, j, :actual_len] = item_data[:actual_len]

                    batch_feat_tensors[k] = torch.from_numpy(batch_data)

                elif k in self.ITEM_EMB_FEAT:
                    # Embedding类型特征
                    emb_dim = self.ITEM_EMB_FEAT[k]
                    batch_data = np.zeros((1, len(current_item_ids), emb_dim), dtype=np.float32)

                    for j, item_id_re_id in enumerate(current_item_ids):
                        if k in feat_dict[item_id_re_id]:
                            batch_data[0, j] = feat_dict[item_id_re_id][k]

                    batch_feat_tensors[k] = torch.from_numpy(batch_data)

                elif k in self.ITEM_CONTINUAL_FEAT or k in self.USER_CONTINUAL_FEAT:
                    # 连续类型特征
                    batch_data = np.zeros((1, len(current_item_ids)), dtype=np.float32)

                    for j, item_id_re_id in enumerate(current_item_ids):
                        if k in feat_dict[item_id_re_id]:
                            batch_data[0, j] = feat_dict[item_id_re_id][k]

                    batch_feat_tensors[k] = torch.from_numpy(batch_data)

                else:
                    # Sparse类型特征
                    batch_data = np.zeros((1, len(current_item_ids)), dtype=np.int64)

                    for j, item_id_re_id in enumerate(current_item_ids):
                        if k in feat_dict[item_id_re_id]:
                            batch_data[0, j] = feat_dict[item_id_re_id][k]

                    batch_feat_tensors[k] = torch.from_numpy(batch_data)

            item_seq_tensor = torch.tensor(current_item_ids, device=self.dev).unsqueeze(0)
            batch_emb = self.feat2emb(item_seq_tensor, batch_feat_tensors, include_user=False).squeeze(0)
            batch_emb_norm = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
            all_embs.append(batch_emb_norm.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))