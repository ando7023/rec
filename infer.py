import argparse
import json
import os
import struct
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel

# PyTorch-based retrieval functions
def pytorch_batch_retrieval(query_embeddings, item_embeddings, item_ids, 
                          top_k=10, batch_size=100, device='cuda'):
    """
    使用PyTorch批量计算最近邻，替代FAISS
    """
    num_queries = query_embeddings.shape[0]
    num_items = item_embeddings.shape[0]
    
    # 将embeddings转换为PyTorch tensors并移到GPU
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    item_tensor = torch.from_numpy(item_embeddings).to(device)
    
    result_ids = []
    
    # 批量处理查询，避免显存溢出
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="PyTorch batch retrieval"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_queries = query_tensor[start_idx:end_idx]
        
        # 为了进一步减少显存使用，可以分块计算相似度
        if num_items > 50000:  # 如果候选集很大，分块计算
            similarities = []
            item_batch_size = 10000
            
            for item_start in range(0, num_items, item_batch_size):
                item_end = min(item_start + item_batch_size, num_items)
                item_batch = item_tensor[item_start:item_end]
                
                # 计算这一批item的相似度
                sim_batch = torch.matmul(batch_queries, item_batch.T)
                similarities.append(sim_batch)
            
            # 合并所有相似度
            similarity = torch.cat(similarities, dim=1)
        else:
            # 直接计算所有相似度
            similarity = torch.matmul(batch_queries, item_tensor.T)
        
        # 找出top-k个最相似的索引
        _, top_indices = torch.topk(similarity, k=top_k, dim=1, largest=True)
        
        # 将索引转换为对应的item IDs
        batch_result_ids = item_ids[top_indices.cpu().numpy()]
        result_ids.append(batch_result_ids.squeeze(-1))
        
        # 清理GPU内存
        del similarity, top_indices
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # 合并所有批次的结果
    result_ids = np.vstack(result_ids)
    
    return result_ids

def load_embeddings(file_path):
    """加载二进制格式的embeddings文件"""
    with open(file_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        num_dimensions = struct.unpack('I', f.read(4))[0]
        embeddings = np.fromfile(f, dtype=np.float32).reshape(num_points, num_dimensions)
    return embeddings

def load_ids(file_path):
    """加载ID文件"""
    with open(file_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        num_dimensions = struct.unpack('I', f.read(4))[0]
        ids = np.fromfile(f, dtype=np.uint64).reshape(num_points, num_dimensions)
    return ids

def save_result_ids(result_ids, file_path, top_k):
    """保存检索结果到文件，格式与FAISS输出兼容"""
    num_queries = result_ids.shape[0]
    
    with open(file_path, 'wb') as f:
        f.write(struct.pack('I', num_queries))
        f.write(struct.pack('I', top_k))
        result_ids.astype(np.uint64).tofile(f)


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    # 遍历目录查找第一个以.pt结尾的文件
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    # 如果没有找到，返回None
    return None


def get_args():
    parser = argparse.ArgumentParser()

    # Train params (部分参数在推理时不直接使用，但模型初始化需要)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)  # 推理时不用
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--temp', default=0.05, type=float, help='Temperature parameter for InfoNCE Loss')

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=12, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)  # 推理时不用
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0, type=float)  # 推理时模型处于eval模式，dropout不生效
    parser.add_argument('--l2_emb', default=0.0, type=float)  # 推理时不用
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)  # 将会被get_ckpt_path覆盖
    #parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    
    # ============ 新增：时间特征相关参数 ============
    parser.add_argument('--use_time_features', default='use_time_features', action='store_true',
                        help='Whether to use time features (only for items to avoid data leakage)')
    parser.add_argument('--time_buckets', type=int, default=15,
                        help='Number of time difference buckets for discretization')
    parser.add_argument('--max_time_diff', type=float, default=86400*30,
                        help='Maximum time difference in seconds (default: 30 days)')
    parser.add_argument('--id_dropout_rate', default=0.3, type=float,
                        help='Probability to apply ID dropout (replace ID with 0).')
    parser.add_argument('--attn_head_dim', default=72, type=int,
                        help='Demension of each attention head for Q and K.')
    parser.add_argument('--linear_head_dim', default=32, type=int,
                        help='Demension of each attention head for U and V.')
    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if isinstance(feat_value, list):  # Use isinstance instead of type == list
            value_list = []
            for v in feat_value:
                if isinstance(v, str):  # Use isinstance
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif isinstance(feat_value, str):  # Use isinstance
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, args):
    """
    生产候选库item的id和embedding
    
    修改：添加args参数以支持时间特征

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
        args: 参数，包含use_time_features等
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')

    all_item_ids = []
    all_retrieval_ids = []
    item_id_to_feature_map = {}
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            
            # 处理特征
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)
            
            # 修改：使用正确的时间特征key
            if args.use_time_features:
                # 使用与dataset.py中一致的key名称
                feature['t_weekday'] = 0  # 不是 '201'
                feature['t_hour'] = 0      # 不是 '202'
                feature['t_diff'] = 0      # 不是 '203'

            all_item_ids.append(item_id)
            all_retrieval_ids.append(retrieval_id)
            item_id_to_feature_map[item_id] = feature
            retrieve_id2creative_id[retrieval_id] = creative_id

    model.save_item_emb(all_item_ids, all_retrieval_ids, item_id_to_feature_map, 
                       os.environ.get('EVAL_RESULT_PATH'))

    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    def set_seed(seed=42):
        """设置所有随机种子以确保可重复性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        
        # 设置CUDNN
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        # 设置Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
    SEED = 7023
    set_seed(SEED)
    
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    
    # ============ 修改：MyTestDataset不再需要cache_path参数 ============
    test_dataset = MyTestDataset(data_path, args)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # Enable multiprocessing for inference as well
        collate_fn=MyTestDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    # The model initialization signature was changed. It no longer takes feature_default_value.
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 获取并加载模型检查点
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(f"No model checkpoint (.pt file) found in {os.environ.get('MODEL_OUTPUT_PATH')}.")

    print(f"Loading model from: {ckpt_path}")
    
    # ============ 新增：处理可能的时间特征不匹配问题 ============
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    
    # 如果checkpoint中没有时间特征的embedding，但args启用了时间特征
    # 需要特殊处理以避免加载错误
    if args.use_time_features:
        model_state = model.state_dict()
        
        # 检查时间特征的embedding是否在checkpoint中
        time_feat_keys = ['sparse_emb.t_weekday.weight', 
                         'sparse_emb.t_hour.weight', 
                         'sparse_emb.t_diff.weight']
        missing_time_feats = [k for k in time_feat_keys if k not in checkpoint]
        
        if missing_time_feats:
            print(f"Warning: Time feature embeddings not found in checkpoint: {missing_time_feats}")
            print("These will be randomly initialized. Consider retraining if time features are important.")
            
            # 只加载匹配的参数
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_state and v.shape == model_state[k].shape}
            model_state.update(pretrained_dict)
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(checkpoint)
    else:
        # 如果不使用时间特征，过滤掉时间相关的参数
        model_state = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                         if k in model_state and v.shape == model_state[k].shape}
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state)

    model.eval()

    all_embs = []  # 用于存储所有用户 embeddings
    user_list = []  # 用于存储用户ID

    # 在推理时禁用梯度计算
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

            seq, token_type, seq_feat, user_id, timestamps= batch

            # 调试：检查时间特征是否存在
            if step == 0:  # 只打印第一个batch
                print("Time features in batch:")
                for time_key in ['t_weekday', 't_hour', 't_diff']:
                    if time_key in seq_feat:
                        unique_vals = torch.unique(seq_feat[time_key])
                        print(f"  {time_key}: shape={seq_feat[time_key].shape}")
                        print(f"    All unique values ({len(unique_vals)} total): {sorted(unique_vals.tolist())}")

            seq = seq.to(args.device, non_blocking=True)
            token_type = token_type.to(args.device, non_blocking=True)
            timestamps = timestamps.to(args.device, non_blocking=True)
            # seq_feat 现在是一个处理好的特征字典，每个value都是tensor
            # 需要将所有tensor移到GPU
            seq_feat_gpu = {}
            for k, v in seq_feat.items():
                seq_feat_gpu[k] = v.to(args.device, non_blocking=True)

            # model.predict 返回的是用户/序列的 embedding (log_feats[:, -1, :])
            user_embs = model.predict(seq, seq_feat_gpu, token_type, timestamps)

            # --- 关键修改：对用户 embeddings 进行 L2 范数归一化 ---
            # 这与 InfoNCE Loss 中计算相似度的方式一致，确保FAISS的内积搜索等同于余弦相似度
            user_embs_norm = user_embs / user_embs.norm(dim=-1, keepdim=True)

            for i in range(user_embs_norm.shape[0]):
                # 将归一化后的 embedding 转换为 NumPy 并添加到列表中
                emb = user_embs_norm[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
                all_embs.append(emb)
            user_list += user_id

    # 合并所有用户 embeddings 并保存为查询文件
    all_embs = np.concatenate(all_embs, axis=0)
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    # 生成候选库的 item embedding 以及 id 文件
    # ============ 修改：传入args参数 ============
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        args  # 新增：传入args以支持时间特征
    )

    # 使用PyTorch进行检索，替代FAISS
    print("Using PyTorch for fast retrieval...")
    
    # 加载embeddings和IDs
    query_embeddings = load_embeddings(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
    item_embeddings = load_embeddings(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
    item_ids = load_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")
    
    # 选择设备
    device = args.device
    print(f"Using device: {device}")
    
    # 执行PyTorch批量检索
    top10s_retrieved = pytorch_batch_retrieval(
        query_embeddings,
        item_embeddings,
        item_ids,
        top_k=10,
        batch_size=128,  # 可以根据GPU显存调整
        device=device
    )
    
    # 保存结果（与FAISS格式兼容）
    save_result_ids(top10s_retrieved, Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"), top_k=10)
    
    print(f"Retrieval completed. Results shape: {top10s_retrieved.shape}")

    # 读取检索结果（这部分代码保持不变）
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved, desc="Processing results"):
        for item in top10:
            # 使用 .get() 方法安全地获取 creative_id，避免 KeyError
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i: i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list
