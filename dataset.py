import json
import pickle
import struct
from pathlib import Path
import os
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集 - 优化版本，在__getitem__中完成所有tensor化
    """

    def __init__(self, data_dir, args, seq_file="seq.jsonl", offset_file="seq_offsets.pkl"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_file_path = self.data_dir / seq_file
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 只在初始化时加载必要的元数据
        with open(Path(self.data_dir, offset_file), 'rb') as f:
            self.seq_offsets = pickle.load(f)

        # 加载特征字典和索引
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            self.indexer = indexer

        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}

        # 初始化特征信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # 预计算所有特征ID集合，避免重复计算
        self.all_feat_ids = set()
        for feat_type in self.feature_types.values():
            self.all_feat_ids.update(feat_type)

    def _init_feat_info(self):
        """初始化特征信息"""
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}

        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116',
        ]
        # ADDED: Time-based features
        feat_types['time_sparse'] = ['t_weekday', 't_hour', 't_diff']

        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        # 初始化默认值
        for feat_id in feat_types['user_sparse'] + feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, []))

        # ADDED: Time feature defaults and statistics
        feat_default_value['t_weekday'] = 0
        feat_statistics['t_weekday'] = 7 + 1  # 7 days + 1 for padding
        feat_default_value['t_hour'] = 0
        feat_statistics['t_hour'] = 24 + 1  # 24 hours + 1 for padding
        feat_default_value['t_diff'] = 0
        feat_statistics['t_diff'] = 15 + 1  # 15 bins + 1 for padding

        for feat_id in feat_types['item_array'] + feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, []))

        for feat_id in feat_types['user_continual'] + feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0

        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        for feat_id in feat_types['item_emb']:
            dim = EMB_SHAPE_DICT.get(feat_id, 32)
            feat_default_value[feat_id] = np.zeros(dim, dtype=np.float32)
            feat_statistics[feat_id] = dim

        return feat_default_value, feat_types, feat_statistics

    def _load_user_data(self, uid):
        """每次打开文件读取，避免多进程问题"""
        with open(self.data_file_path, 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """负采样"""
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _get_time_features_with_diff(self, timestamp, time_diff_seconds):
        """使用时间差（秒）计算时间特征"""
        if timestamp == 0:
            return {'t_weekday': 0, 't_hour': 0, 't_diff': 0}
        
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        
        # 计算时间差的桶（使用15个桶）
        time_diff_days = time_diff_seconds / 86400 if time_diff_seconds > 0 else 0
        
        if time_diff_days <= 0:
            diff_bin = 0  # 用0表示没有时间差信息
        elif time_diff_days < 0.5:
            diff_bin = 1  # <12小时
        elif time_diff_days < 1:
            diff_bin = 2  # 12-24小时
        elif time_diff_days < 2:
            diff_bin = 3  # 1-2天
        elif time_diff_days < 3:
            diff_bin = 4  # 2-3天
        elif time_diff_days < 5:
            diff_bin = 5  # 3-5天
        elif time_diff_days < 7:
            diff_bin = 6  # 5-7天
        elif time_diff_days < 14:
            diff_bin = 7  # 1-2周
        elif time_diff_days < 21:
            diff_bin = 8  # 2-3周
        elif time_diff_days < 30:
            diff_bin = 9  # 3-4周
        elif time_diff_days < 45:
            diff_bin = 10  # 1-1.5个月
        elif time_diff_days < 60:
            diff_bin = 11  # 1.5-2个月
        elif time_diff_days < 90:
            diff_bin = 12  # 2-3个月
        elif time_diff_days < 180:
            diff_bin = 13  # 3-6个月
        elif time_diff_days < 365:
            diff_bin = 14  # 6-12个月
        else:
            diff_bin = 15  # >1年

        return {
            't_weekday': dt.weekday() + 1,
            't_hour': dt.hour + 1,
            't_diff': diff_bin
        }

    def fill_missing_feat(self, feat, item_id, time_feats=None):
        """填充缺失特征"""
        if feat is None:
            feat = {}

        filled_feat = feat.copy()

        # 填充缺失的特征
        missing_fields = self.all_feat_ids - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]

        # ADDED: Add time features
        if time_feats:
            filled_feat.update(time_feats)

        # 处理多模态embedding特征
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev.get(item_id) in self.mm_emb_dict.get(feat_id, {}):
                emb = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
                if isinstance(emb, np.ndarray):
                    filled_feat[feat_id] = emb

        return filled_feat

    def _process_sequence_to_tensors(self, seq_feat_list):
        """
        将特征列表转换为tensor字典
        """
        feat_tensors = {}
        all_sparse_feats = self.feature_types['user_sparse'] + self.feature_types['item_sparse'] + self.feature_types[
            'time_sparse']

        for feat_id in all_sparse_feats:
            feat_data = np.zeros(self.maxlen + 1, dtype=np.int64)
            for idx, feat_dict in enumerate(seq_feat_list):
                if feat_dict is not None and feat_id in feat_dict:
                    feat_data[idx] = feat_dict[feat_id]
            feat_tensors[feat_id] = feat_data

        for feat_id in self.feature_types['user_array'] + self.feature_types['item_array']:
            max_len = 1
            for feat_dict in seq_feat_list:
                if feat_dict is not None and feat_id in feat_dict:
                    max_len = max(max_len, len(feat_dict[feat_id]))

            feat_data = np.zeros((self.maxlen + 1, max_len), dtype=np.int64)
            for idx, feat_dict in enumerate(seq_feat_list):
                if feat_dict is not None and feat_id in feat_dict:
                    arr = feat_dict[feat_id]
                    feat_data[idx, :len(arr)] = arr[:max_len]
            feat_tensors[feat_id] = feat_data

        for feat_id in self.feature_types['item_emb']:
            emb_dim = self.feat_statistics[feat_id]
            feat_data = np.zeros((self.maxlen + 1, emb_dim), dtype=np.float32)
            for idx, feat_dict in enumerate(seq_feat_list):
                if feat_dict is not None and feat_id in feat_dict:
                    feat_data[idx] = feat_dict[feat_id]
            feat_tensors[feat_id] = feat_data

        for feat_id in self.feature_types['user_continual'] + self.feature_types['item_continual']:
            feat_data = np.zeros(self.maxlen + 1, dtype=np.float32)
            for idx, feat_dict in enumerate(seq_feat_list):
                if feat_dict is not None and feat_id in feat_dict:
                    feat_data[idx] = feat_dict[feat_id]
            feat_tensors[feat_id] = feat_data

        return feat_tensors

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
    
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

        # 预计算每个位置距离上一个交互的时间差
        time_gaps = [0] * len(ext_user_sequence)
        for i in range(1, len(ext_user_sequence)):
            prev_time = ext_user_sequence[i-1][4] if len(ext_user_sequence[i-1]) > 4 else 0
            curr_time = ext_user_sequence[i][4] if len(ext_user_sequence[i]) > 4 else 0
            if prev_time > 0 and curr_time > 0:
                time_gaps[i] = curr_time - prev_time  # 当前交互距离上一个交互的时间
            else:
                time_gaps[i] = 0
        
        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        pos = np.zeros(self.maxlen + 1, dtype=np.int32)
        neg = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_action_type = np.zeros(self.maxlen + 1, dtype=np.int32)

        # 【新增】: 创建一个数组用于存储原始时间戳
        timestamps_arr = np.zeros(self.maxlen + 1, dtype=np.int64)

        seq_feat_list = [None] * (self.maxlen + 1)
        pos_feat_list = [None] * (self.maxlen + 1)
        neg_feat_list = [None] * (self.maxlen + 1)

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        nxt = ext_user_sequence[-1] if ext_user_sequence else None
        nxt_idx = len(ext_user_sequence) - 1
        idx = self.maxlen

        for enum_idx, record_tuple in enumerate(reversed(ext_user_sequence[:-1])):
            if idx < 0:
                break

            curr_idx = len(ext_user_sequence) - 2 - enum_idx  # 在原始序列中的索引
            
            i, feat, type_, act_type, timestamp = record_tuple
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt

            # 获取预计算的时间差
            curr_time_gap = time_gaps[curr_idx]  # 当前item距离它前一个交互的时间
            next_time_gap = time_gaps[nxt_idx]   # next item距离它前一个交互的时间
            
            # 使用时间差计算时间特征
            time_feats = self._get_time_features_with_diff(timestamp, curr_time_gap)
            next_time_feats = self._get_time_features_with_diff(next_timestamp, next_time_gap)

            feat = self.fill_missing_feat(feat, i, time_feats)
            next_feat = self.fill_missing_feat(next_feat, next_i, next_time_feats)

            seq[idx] = i

            timestamps_arr[idx] = timestamp

            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            seq_feat_list[idx] = feat

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat_list[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                # 负样本使用相同的时间上下文
                neg_feat_list[idx] = self.fill_missing_feat(
                    self.item_feat_dict.get(str(neg_id), {}), neg_id, next_time_feats
                )

            nxt = record_tuple
            nxt_idx = curr_idx
            idx -= 1

        # 填充默认值
        for idx in range(self.maxlen + 1):
            if seq_feat_list[idx] is None:
                seq_feat_list[idx] = self.feature_default_value.copy()
            if pos_feat_list[idx] is None:
                pos_feat_list[idx] = self.feature_default_value.copy()
            if neg_feat_list[idx] is None:
                neg_feat_list[idx] = self.feature_default_value.copy()

        seq_feat_tensors = self._process_sequence_to_tensors(seq_feat_list)
        pos_feat_tensors = self._process_sequence_to_tensors(pos_feat_list)
        neg_feat_tensors = self._process_sequence_to_tensors(neg_feat_list)

        return (seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, timestamps_arr)
    def __len__(self):
        return len(self.seq_offsets)

    # ... (collate_fn and other methods remain the same) ...
    @staticmethod
    def collate_fn(batch):
        """
        collate_fn只负责组装batch，不做复杂处理
        因为tensor化已经在__getitem__中完成
        """
        (seq_list, pos_list, neg_list, token_type_list, next_token_type_list,
         next_action_type_list, seq_feat_list, pos_feat_list, neg_feat_list,
     timestamps_list) = zip(*batch)

        batch_size = len(seq_list)

        # 简单stack基础数据
        seq = torch.from_numpy(np.stack(seq_list))
        pos = torch.from_numpy(np.stack(pos_list))
        neg = torch.from_numpy(np.stack(neg_list))
        token_type = torch.from_numpy(np.stack(token_type_list))
        next_token_type = torch.from_numpy(np.stack(next_token_type_list))
        next_action_type = torch.from_numpy(np.stack(next_action_type_list))

        timestamps = torch.from_numpy(np.stack(timestamps_list))

        # 组装特征tensor字典
        seq_feat_batch = {}
        pos_feat_batch = {}
        neg_feat_batch = {}

        # 获取所有特征keys（从第一个样本）
        all_keys = seq_feat_list[0].keys()

        for k in all_keys:
            # 收集该特征的所有样本
            seq_arrays = [seq_feat_list[i][k] for i in range(batch_size)]
            pos_arrays = [pos_feat_list[i][k] for i in range(batch_size)]
            neg_arrays = [neg_feat_list[i][k] for i in range(batch_size)]

            # 检查是否需要padding（array和emb特征）
            if len(seq_arrays[0].shape) == 2:
                # 找最大维度
                max_dim = max(arr.shape[1] for arr in seq_arrays)

                # padding到相同维度
                seq_padded = []
                pos_padded = []
                neg_padded = []

                for i in range(batch_size):
                    # 创建padded数组
                    seq_pad = np.zeros((seq_arrays[i].shape[0], max_dim), dtype=seq_arrays[i].dtype)
                    pos_pad = np.zeros((pos_arrays[i].shape[0], max_dim), dtype=pos_arrays[i].dtype)
                    neg_pad = np.zeros((neg_arrays[i].shape[0], max_dim), dtype=neg_arrays[i].dtype)

                    # 复制原始数据
                    seq_pad[:, :seq_arrays[i].shape[1]] = seq_arrays[i]
                    pos_pad[:, :pos_arrays[i].shape[1]] = pos_arrays[i]
                    neg_pad[:, :neg_arrays[i].shape[1]] = neg_arrays[i]

                    seq_padded.append(seq_pad)
                    pos_padded.append(pos_pad)
                    neg_padded.append(neg_pad)

                seq_feat_batch[k] = torch.from_numpy(np.stack(seq_padded))
                pos_feat_batch[k] = torch.from_numpy(np.stack(pos_padded))
                neg_feat_batch[k] = torch.from_numpy(np.stack(neg_padded))
            else:
                # 1D特征，直接stack
                seq_feat_batch[k] = torch.from_numpy(np.stack(seq_arrays))
                pos_feat_batch[k] = torch.from_numpy(np.stack(pos_arrays))
                neg_feat_batch[k] = torch.from_numpy(np.stack(neg_arrays))

        return (seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat_batch, pos_feat_batch, neg_feat_batch, timestamps)


# ... (The rest of the file remains the same) ...
class MyTestDataset(MyDataset):
    """测试数据集"""

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args,
                         seq_file="predict_seq.jsonl",
                         offset_file="predict_seq_offsets.pkl")

        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """处理冷启动特征"""
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                value_list = []
                for v in feat_value:
                    value_list.append(0 if isinstance(v, str) else v)
                processed_feat[feat_id] = value_list
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat
    

    def __getitem__(self, uid):
        """测试集的__getitem__也要完成tensor化"""
        user_sequence = self._load_user_data(uid)

        timestamps_arr = np.zeros(self.maxlen + 1, dtype=np.int64)

        # 调试：检查原始数据的timestamp
        if not hasattr(self, '_debug_done'):
            self._debug_done = True
            print(f"\n=== Test Data Debug for uid {uid} ===")
            for i, record in enumerate(user_sequence[:3]):  # 只看前3条
                if len(record) > 5:
                    print(f"Record {i}: timestamp={record[5]}")
                else:
                    print(f"Record {i}: NO TIMESTAMP")

        
        ext_user_sequence = []
        user_id = None
        
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = (
                record_tuple[:6] if len(record_tuple) >= 6 
                else (*record_tuple[:5], 0)
            )
            
            if u:
                if isinstance(u, str):
                    user_id = u
                    u = 0
                else:
                    user_id = self.indexer_u_rev.get(u, f"user_{u}")
                
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                    ext_user_sequence.insert(0, (u, user_feat, 2, timestamp))
            
            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                    ext_user_sequence.append((i, item_feat, 1, timestamp))

        # 预计算时间差
        time_gaps = [0] * len(ext_user_sequence)
        for i in range(1, len(ext_user_sequence)):
            prev_time = ext_user_sequence[i-1][3] if len(ext_user_sequence[i-1]) > 3 else 0
            curr_time = ext_user_sequence[i][3] if len(ext_user_sequence[i]) > 3 else 0
            if prev_time > 0 and curr_time > 0:
                time_gaps[i] = curr_time - prev_time
            else:
                time_gaps[i] = 0

        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        seq_feat_list = [None] * (self.maxlen + 1)

        idx = self.maxlen
        
        for enum_idx, record_tuple in enumerate(reversed(ext_user_sequence)):
            if idx < 0:
                break
            
            curr_idx = len(ext_user_sequence) - 1 - enum_idx
            
            i_id, feat, type_, timestamp = (
                record_tuple[:4] if len(record_tuple) >= 4 
                else (*record_tuple[:3], 0)
            )
            
            # 使用预计算的时间差
            curr_time_gap = time_gaps[curr_idx]
            time_feats = self._get_time_features_with_diff(timestamp, curr_time_gap)
            feat = self.fill_missing_feat(feat, i_id, time_feats)
            
            seq[idx] = i_id

            timestamps_arr[idx] = timestamp

            token_type[idx] = type_
            seq_feat_list[idx] = feat
            
            idx -= 1

        # 填充默认值
        for idx in range(self.maxlen + 1):
            if seq_feat_list[idx] is None:
                seq_feat_list[idx] = self.feature_default_value.copy()

        seq_feat_tensors = self._process_sequence_to_tensors(seq_feat_list)
        
        return seq, token_type, seq_feat_tensors, user_id, timestamps_arr

    def __len__(self):
        return len(self.seq_offsets)

    @staticmethod
    def collate_fn(batch):
        """测试集的collate_fn"""
        seq_list, token_type_list, seq_feat_list, user_id_list, timestamps_list = zip(*batch)

        batch_size = len(seq_list)

        seq = torch.from_numpy(np.stack(seq_list))
        token_type = torch.from_numpy(np.stack(token_type_list))

        timestamps = torch.from_numpy(np.stack(timestamps_list))

        # 组装特征tensor字典
        seq_feat_batch = {}
        all_keys = seq_feat_list[0].keys()

        for k in all_keys:
            seq_arrays = [seq_feat_list[i][k] for i in range(batch_size)]

            if len(seq_arrays[0].shape) == 2:
                max_dim = max(arr.shape[1] for arr in seq_arrays)
                seq_padded = []

                for i in range(batch_size):
                    seq_pad = np.zeros((seq_arrays[i].shape[0], max_dim), dtype=seq_arrays[i].dtype)
                    seq_pad[:, :seq_arrays[i].shape[1]] = seq_arrays[i]
                    seq_padded.append(seq_pad)

                seq_feat_batch[k] = torch.from_numpy(np.stack(seq_padded))
            else:
                seq_feat_batch[k] = torch.from_numpy(np.stack(seq_arrays))

        return seq, token_type, seq_feat_batch, user_id_list, timestamps


def save_emb(emb, save_path):
    """保存Embedding"""
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """加载多模态特征Embedding"""
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}

    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}

        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        else:
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)

        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')

    return mm_emb_dict