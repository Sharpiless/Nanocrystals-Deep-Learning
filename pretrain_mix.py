import warnings

warnings.filterwarnings('ignore')
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import torch_geometric.nn as geom_nn
import math
import random
import cv2
import os
from models.model import load_weights, HybridModelV4, HybridModelV5, ModelEMA
from utils.utils import seed_everything
from utils.eval_utils import evaluate_model_all
from utils.data_utils import CustomDataset, read_csv_database
import pickle as pkl

import logging
import os
import torch.nn.init as init
    
# 自定义初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
 
def adjust_learning_rate(optimizer, epoch, iter, total_iters):
    if iter < warmup_iters:
        lr = warmup_factor + (initial_lr - warmup_factor) * iter / warmup_iters
    else:
        lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters))) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_logger(save_path):
    logger = logging.getLogger("debug")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
    
if __name__ == '__main__':
    from utils.flags_set import get_args
    args = get_args()
    
    with open("data/shapes_list.txt", "r") as f:
        labels = f.read().splitlines()
    labels.append("fail")
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    exp_dir = f"{args.save_path}/results-{args.exp}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    # encoding='utf-8'
        
    seed_everything(args.seed)

    # Load the dataset
    conditions = ['Temperature1', 'Temperature2', 'Time', 'Speed']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load logger
    logger = get_logger(os.path.join(exp_dir, "log.txt"))
    logger.info(args)
    
    # Load scalers
    if args.scalers_path:
        logger.info("- scaler - loading scalers from: {args.scalers_path}")
        with open(args.scalers_path, "rb") as f:
            scalers_dict = pkl.load(f)
    else:
        scalers_dict = None

    # Model parameters
    if args.loss == "mse":
        criterion_reg = nn.MSELoss(reduce=False)
    elif args.loss == "mae":
        criterion_reg = nn.L1Loss(reduce=False)
    elif args.loss == "huber":
        criterion_reg = nn.HuberLoss(delta=1.0, reduction="none")
    else:
        raise NotImplementedError
    criterion_cls = nn.CrossEntropyLoss()

    # K-Fold Cross Validation
    fold_results = []
    
    # model to use
    if args.model_version == "v4":
        model2use = HybridModelV4
    elif args.model_version == "v5":
        model2use = HybridModelV5
    else:
        logger.info("- model - args.model_version: {args.model_version}")
        raise NotImplementedError
    
    eval_results = {"mse": [], "rmse": [], "mape": [], "mae": [], "r2": []}
    
    for fold in range(5):
        train_dataset = CustomDataset(
            args, fold, True, args.raw_data, conditions, args.max_threshold, scalers_dict, logger
        )
        
        tmp_results = {"mse": 100, "rmse": 100, "mape": 100, "mae": 100, "r2": 0.0}
        model = model2use(
            args.model_type, 
            args.cls_token,
            args.num_classes+1,
            args.input_size+len(conditions), 
            args.hidden_size, 
            args.dim_feedforward,
            args.num_layers, 
            args.num_head,
            len(conditions),
            average_feats=args.average_feats,
            with_bn=args.with_bn,
            dropout_rate=args.dropout)
        
        if not args.without_kaiming_normal:
            model.apply(init_weights)
        
        if args.pretrained:
            logger.info(f"- model - loading from: {args.pretrained}/best-{fold}.pth")
            load_weights(model, f"{args.pretrained}/best-{fold}.pth")
            
        if args.frozen:
            for name, p in model.named_parameters():
                for i in range(args.frozen):
                    if f"transformer.layers.{i}" in name:
                        print("frozen:", name)
                        p.requires_grad = False
                if name in ["cls_token", "embedding.weight", "embedding.bias"]:
                    print("frozen:", name)
                    p.requires_grad = False
        
        model = model.to(device)
        ema_model = ModelEMA(model, decay=0.99)
        
        # data to train
        train_data_dict, val_data_dict = {}, {}
        for k, v in train_dataset.data_dict.items():
            train_data_dict[k] = v[train_dataset.train_idx]
            val_data_dict[k] = v[train_dataset.val_idx]
        
        # val dataset
        X_train = torch.tensor(train_data_dict["X_features"], dtype=torch.float).to(device)
        C_train = torch.tensor(train_data_dict["C_scaled"], dtype=torch.float).to(device)
        mask_train = torch.tensor(train_data_dict["mask"], dtype=torch.bool).to(device)
        y_reg_size_train = torch.tensor(train_data_dict["y_reg_size"], dtype=torch.float).to(device)
        y_cls_train = torch.tensor(train_data_dict["y_cls"], dtype=torch.float).to(device)
        logger.info(f"- data - origin train size: {X_train.shape[0]}")
        
        if args.local_expand_data:
            expand_data = f"extra_data_all/fold{fold}-10000.csv"
            exp_data_dict, _, _ = read_csv_database(
                args, expand_data, conditions, 
                args.max_threshold, train_dataset.obj_features,
                train_dataset.scalers_dict, logger=logger, sample_ratio=0.5
            )
            exp_num = exp_data_dict["nums"].shape[0]
            if args.not_trust_extra:
                exp_data_dict["nums"][:] = -1
            logger.info(f"- data - loading expaned data: {exp_num}")
            for k, v in train_data_dict.items():
                train_data_dict[k] = np.concatenate([v, exp_data_dict[k]])
        
        if args.load_failure:
            # expand_data = f"data/0309_failure.csv"
            expand_data = f"data/0404_failure.csv"
            exp_data_dict, _, _ = read_csv_database(
                args, expand_data, conditions, 
                args.max_threshold, train_dataset.obj_features,
                train_dataset.scalers_dict, logger=logger
            )
            exp_num = exp_data_dict["nums"].shape[0]
            logger.info(f"- data - loading failure data: {exp_num}")
            for k, v in train_data_dict.items():
                train_data_dict[k] = np.concatenate([v, exp_data_dict[k]])
        
        train_dataset_torch = torch.utils.data.TensorDataset(
            torch.tensor(train_data_dict["X_features"], dtype=torch.float).to(device),
            torch.tensor(train_data_dict["C_scaled"], dtype=torch.float).to(device),
            torch.tensor(train_data_dict["mask"], dtype=torch.bool).to(device),
            torch.tensor(train_data_dict["y_reg_size"], dtype=torch.float).to(device),
            torch.tensor(train_data_dict["y_reg_dist"], dtype=torch.float).to(device),
            torch.tensor(train_data_dict["y_cls"], dtype=torch.int64).to(device),
            torch.tensor(train_data_dict["nums"], dtype=torch.float).to(device)
        )
        train_loader = DataLoader(dataset=train_dataset_torch, batch_size=args.batch_size, shuffle=True)
        
        # val dataset
        X_val = torch.tensor(val_data_dict["X_features"], dtype=torch.float).to(device)
        C_val = torch.tensor(val_data_dict["C_scaled"], dtype=torch.float).to(device)
        mask_val = torch.tensor(val_data_dict["mask"], dtype=torch.bool).to(device)
        y_reg_size_val = torch.tensor(val_data_dict["y_reg_size"], dtype=torch.float).to(device)
        y_cls_val = torch.tensor(val_data_dict["y_cls"], dtype=torch.float).to(device)
        logger.info(f"- data - train size:{X_train.shape[0]} val size: {X_val.shape[0]}")

        # Loss and optimizer
        initial_lr = args.lr / 5
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)
        # Total number of epochs and iterations
        total_iters = args.epoch * len(train_loader)  # total number of iterations

        # Warmup settings
        warmup_epochs = 10
        warmup_iters = warmup_epochs * len(train_loader)
        warmup_factor = initial_lr / 10
        min_lr = initial_lr / 10
        
        # Train the model
        with tqdm(total=args.epoch) as t:
            for epoch in range(args.epoch):  # Number of epochs can be adjusted
                if args.dropout > 0:
                    model.train() # 开启 dropout
                else:
                    model.eval() # 不开启 dropout
                    
                loss_list = []
                for i, (x_inputs, c_inputs, x_mask, y_reg_size, y_reg_dist, y_cls, x_num) in enumerate(train_loader):
                    y_reg_size = y_reg_size.unsqueeze(-1)
                    x_num = x_num.unsqueeze(-1)
                    current_iter = epoch * len(train_loader) + i
                    adjust_learning_rate(optimizer, epoch, current_iter, total_iters)
                    if args.mask_target and epoch < int(args.pretrain_ratio * args.epoch):
                        mask_rate = args.mask_target_rate / 2
                        rand_values = torch.rand((x_inputs.shape[0], ))
                        rand_mask1 = rand_values < mask_rate
                        rand_mask2 = rand_values > 1 - mask_rate
                        x_inputs[:, 0, 2:][rand_mask1] = 0.0
                        rand_mask = torch.rand((x_inputs.shape[0], x_inputs.shape[1]))
                        x_inputs[
                            torch.arange(x_inputs.shape[0])[rand_mask2], 
                            rand_mask.argmax(-1)[rand_mask2], 2:
                        ] = 0.0
                    x_inputs = torch.cat([x_inputs, c_inputs[:, None, :].repeat(1, x_inputs.shape[1], 1)], -1)
                    # Forward pass
                    outputs_size, outputs_dist, outputs_cls = model(x_inputs, c_inputs, x_mask)
                    # if args.without_size:
                    if not True:
                        loss = 0.0
                    else:
                        loss = criterion_reg(outputs_size, y_reg_size)
                        if args.weak_data: 
                            mape = torch.abs(outputs_size - y_reg_size).detach() / y_reg_size.detach()
                            weak_mask = torch.logical_and(mape <= args.weak_threshold, y_reg_size > 0)
                            weak_mask = torch.logical_and(weak_mask, x_num < args.num_threshold)
                            solid_mask = 1 - weak_mask.float()
                            loss = loss * solid_mask
                            loss = loss.sum() / solid_mask.sum()
                        else:
                            loss = loss.mean()
                        
                    # if not args.without_cls:
                    if True:
                        loss += criterion_cls(outputs_cls, y_cls) * 5.0
                    # if not args.without_dist:
                    if not True:
                        loss += criterion_reg(outputs_dist[:, 0], y_reg_dist).mean() * 0.2
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ema_model.update(model)
                    loss_list.append(loss.detach().cpu().numpy())
                
                t.set_description(f"epoch: {epoch}")
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                t.set_postfix(loss=np.mean(loss_list), lr=lr, iter=f"{current_iter}/{total_iters}")
                t.update(1)

                if (epoch + 1) % 50 == 0 or epoch == args.epoch - 1:
                    # Visualization model results
                    logger.info("- eval - evaluating train set (ema) ...")
                    evaluate_model_all(
                        None, model, X_train, C_train, mask_train, y_reg_size_train, y_cls_train, labels, logger
                    )
                    # Visualization ema model results
                    logger.info("- eval - evaluating val set (ema) ...")
                    fold_eval_results_ema, _, _, is_best = evaluate_model_all(
                        tmp_results, ema_model.ema_model, X_val, C_val, 
                        mask_val, y_reg_size_val, y_cls_val, labels, logger
                    )
                    # fold_eval_results = fold_eval_results_ema
                    mape_ema = fold_eval_results_ema['mape']
                    r2_ema = fold_eval_results_ema['r2']
                    mae_ema = fold_eval_results_ema['mae']
                    if is_best:
                        logger.info(f"- eval - fold: {fold}, epoch: {epoch}, mae (ema): {mae_ema:.4f}")
                        logger.info(f"- eval - fold: {fold}, epoch: {epoch}, mape (ema): {mape_ema:.4f}")
                        logger.info(f"- eval - fold: {fold}, epoch: {epoch}, r2 (ema): {r2_ema:.4f}")
                        logger.info(f"- eval - save checkpoint to: {exp_dir}/best-{fold}.pth")
                        state_dict = {
                            "model_ema": ema_model.ema_model.state_dict(),
                            "fold": fold
                        }
                        torch.save(state_dict, f"{exp_dir}/best-{fold}.pth")
                    state_dict = {
                        "model_ema": ema_model.ema_model.state_dict(),
                        "fold": fold
                    }
                    torch.save(state_dict, f"{exp_dir}/latest-{fold}.pth")
            
        for metric, value in fold_eval_results_ema.items():
            eval_results[metric].append(value)
            
        with open(f"{args.save_path}/{args.exp}.txt", "a") as f:
            mape_ema = fold_eval_results_ema['mape']
            r2_ema = fold_eval_results_ema['r2']
            mae_ema = fold_eval_results_ema['mae']
            f.write(f"fold: {fold}, mae (ema): {mae_ema:.4f}, mape (ema): {mape_ema:.4f}, r2 (ema): {r2_ema:.4f}\n")
            logger.info(f"fold: {fold}, mae (ema): {mae_ema:.4f}")
            logger.info(f"fold: {fold}, mape (ema): {mape_ema:.4f}")
            logger.info(f"fold: {fold}, r2 (ema): {r2_ema:.4f}")
                    
    with open(f"{args.save_path}/{args.exp}.txt", "a") as f:
        f.write("average results:\n")
        for metric in eval_results:
            mean_value = np.mean(eval_results[metric])
            logger.info(f"- eval - {metric}: {mean_value:.4f}")
            f.write(f"{metric}: {mean_value:.4f}\n")
        f.write("\n")
        f.write(str(args))
        f.write("\n")