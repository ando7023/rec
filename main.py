import argparse
import json
import os
import time
from pathlib import Path
import random
import numpy as np
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
import torch.cuda.amp as amp  # 确保已导入
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel  # 确保 BaselineModel 已经按前述指引修改


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--temp', default=0.05, type=float,
                        help='Temperature parameter for InfoNCE Loss')

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=12, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)  # weight_decay
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--attn_head_dim', default=72, type=int,
                        help='Demension of each attention head for Q and K.')
    parser.add_argument('--linear_head_dim', default=32, type=int,
                        help='Demension of each attention head for U and V.')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # 【新增】: 添加ID Dropout的命令行参数
    parser.add_argument('--id_dropout_rate', default=0.2, type=float,
                        help='Probability to apply ID dropout (replace ID with 0).')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

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

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=MyDataset.collate_fn,
    pin_memory=True,
    persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=MyDataset.collate_fn, 
    pin_memory=True,
    persistent_workers=True
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # for name, param in model.named_parameters():
    #     try:
    #         torch.nn.init.xavier_normal_(param.data)
    #     except Exception:
    #         pass

    # model.pos_emb.weight.data[0, :] = 0
    # model.item_emb.weight.data[0, :] = 0
    # model.user_emb.weight.data[0, :] = 0
    for name, param in model.named_parameters():
        if 'item_emb.weight' in name or 'user_emb.weight' in name:
            # Zero初始化 user/item embeddings（专家建议）
            torch.nn.init.zeros_(param.data)
    
        # elif 'pos_emb.weight' in name:
        #     # 位置编码也零初始化
        #     torch.nn.init.zeros_(param.data)
        elif 'sparse_emb' in name and 'weight' in name:
            # 稀疏特征embedding用小的随机初始化
            torch.nn.init.normal_(param.data, mean=0.0, std=0.01)
        elif 'layernorm' in name.lower():
            if 'weight' in name:
                torch.nn.init.ones_(param.data)
            elif 'bias' in name:
                torch.nn.init.zeros_(param.data)
        elif 'weight' in name and len(param.shape) >= 2:
            # 其他权重矩阵保持Xavier
            torch.nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            torch.nn.init.zeros_(param.data)

    # 确保padding位置是0（这部分保留）
    # model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0



    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    warmup_steps = int(0.1 * len(train_loader) * args.num_epochs)
    total_steps = len(train_loader) * args.num_epochs


    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress_after_warmup = max(0.0, float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + np.cos(np.pi * progress_after_warmup))


    #scheduler = LambdaLR(optimizer, lr_lambda)

    scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=len(train_loader),  # 第一个周期的长度
    T_mult=2,  # 每个周期长度的倍数
    eta_min=1e-6
    )

    # 推荐使用新的 GradScaler 初始化方式，避免 FutureWarning
    scaler = torch.amp.GradScaler(enabled=(args.device == 'cuda'))  # 只有在CUDA设备上才启用混合精度

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, timestamps = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            timestamps = timestamps.to(args.device)

            optimizer.zero_grad()  # 每次迭代开始前清空梯度

            # 使用新的 autocast 初始化方式，避免 FutureWarning
            with torch.amp.autocast(device_type=args.device):
                seq_embs, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat,
                    timestamps
                )
                # 解决 UserWarning: To copy construct from a tensor...
                # 确保 next_token_type 是张量并且在正确设备上
                loss_mask = (next_token_type.to(args.device) == 1) if isinstance(next_token_type, torch.Tensor) else (
                            torch.tensor(next_token_type, device=args.device) == 1)
                loss = model.compute_infnce_loss(seq_embs, pos_embs, neg_embs, loss_mask=loss_mask)

            # --- 唯一的反向传播和参数更新 ---
            scaler.scale(loss).backward()  # 计算梯度并缩放

            # (可选) 梯度裁剪，如果需要：
            scaler.unscale_(optimizer) # 在裁剪前需要unscale梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm的值可以调整

            scaler.step(optimizer)  # 根据缩放后的梯度更新模型参数
            scaler.update()  # 更新 scaler 状态，处理 Inf/NaN 梯度等

            scheduler.step()  # 更新学习率

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1
            
        # 验证循环
        model.eval()
        valid_loss_sum = 0
        # 验证时也应该禁用梯度计算
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, timestamps = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                timestamps = timestamps.to(args.device)

                # 验证时也使用 autocast (可选，但推荐保持一致性)
                with torch.amp.autocast(device_type=args.device):
                    seq_embs, pos_embs, neg_embs = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat,
                        timestamps
                    )
                    # 解决 UserWarning: To copy construct from a tensor...
                    loss_mask = (next_token_type.to(args.device) == 1) if isinstance(next_token_type,
                                                                                     torch.Tensor) else (
                                torch.tensor(next_token_type, device=args.device) == 1)
                    loss = model.compute_infnce_loss(seq_embs, pos_embs, neg_embs, loss_mask=loss_mask)

                valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()

