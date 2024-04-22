import torch
import pandas as pd
import pickle
import numpy as np
from utils.utils import calculate_mae, calculate_r2, calculate_mape
from utils.utils import calculate_mse, calculate_rmse
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model_cls(tmp_results, model, X_val, C_val, mask_val, y_cls_val, labels, logger, load_failure):
    # Evaluate the model
    model.eval()
    is_best = False
    with torch.no_grad():
        inputs = torch.cat([X_val, C_val[:, None, :].repeat(1, X_val.shape[1], 1)], -1)
        _, _, y_pred_cls = model(inputs, C_val, mask_val)
        if load_failure:
            y_pred_cls = y_pred_cls[:, :-1]
        y_pred_cls = np.argmax(y_pred_cls.cpu().numpy(), -1)
        y_true_cls = y_cls_val.cpu().numpy().astype(np.int64)
        unique_indexes = sorted(np.unique(np.concatenate([y_true_cls, y_pred_cls])).tolist())
        
        labels2show = [labels[i] for i in unique_indexes]
        res = classification_report(y_true_cls, y_pred_cls, target_names=labels2show)
        
        acc = accuracy_score(y_true_cls, y_pred_cls)
        
        # 计算精确率
        precision = precision_score(y_true_cls, y_pred_cls, average='weighted')
        # 计算召回率
        recall = recall_score(y_true_cls, y_pred_cls, average='weighted')
        # 计算F1分数
        f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')
        
        # Calculate metrics
        if not tmp_results is None:
            is_best = acc >= tmp_results["accuracy"]
            if is_best:
                tmp_results["accuracy"] = acc
                tmp_results["precision"] = precision
                tmp_results["recall"] = recall
                tmp_results["f1"] = f1
    
    # if not tmp_results is None:
        # print(res)
    logger.info(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    return tmp_results, y_true_cls, y_pred_cls, is_best

def evaluate_model(tmp_results, model, X_val, C_val, mask_val, y_reg_val, logger, clip=None):
    # Evaluate the model
    model.eval()
    is_best = False
    with torch.no_grad():
        inputs = torch.cat([X_val, C_val[:, None, :].repeat(1, X_val.shape[1], 1)], -1)
        y_pred_reg, _, _ = model(inputs, C_val, mask_val)
        y_pred_reg = y_pred_reg.cpu().numpy()[:, 0]
        if clip:
            y_pred_reg = np.clip(y_pred_reg, 0, clip)
        y_true_reg = y_reg_val.cpu().numpy()
        # Calculate metrics
        mape = calculate_mape(y_true_reg, y_pred_reg)
        r2 = calculate_r2(y_true_reg, y_pred_reg)
        mae = calculate_mae(y_true_reg, y_pred_reg)
        if not tmp_results is None:
            is_best = mae <= tmp_results["mae"]
            if is_best:
                tmp_results["mse"] = calculate_mse(y_true_reg, y_pred_reg)
                tmp_results["rmse"] = calculate_rmse(y_true_reg, y_pred_reg)
                tmp_results["mape"] = mape
                tmp_results["mae"] = mae
                tmp_results["r2"] = r2
    logger.info(f"mae: {mae:.4f}, mape: {mape:.4f}, r2: {r2:.4f}")
    return tmp_results, y_true_reg, y_pred_reg, is_best

def evaluate_model_vis(fold, tmp_results, model, X_val, C_val, mask_val, y_reg_val, device, index):
    # Evaluate the model
    model.eval()
    is_best = False
    with torch.no_grad():
        inputs = torch.cat([X_val, C_val[:, None, :].repeat(1, X_val.shape[1], 1)], -1)
        _, _, _, attn_weights = model(
            inputs.to(device)[index].unsqueeze(0), 
            C_val.to(device)[index].unsqueeze(0), 
            mask_val.to(device)[index].unsqueeze(0), return_attn=True
        )
    return attn_weights

def evaluate_model_all(tmp_results, model, X_val, C_val, mask_val, y_reg_val, y_cls_val, labels, logger, y_dist_val=None):
    # Evaluate the model
    model.eval()
    is_best = False
    with torch.no_grad():
        inputs = torch.cat([X_val, C_val[:, None, :].repeat(1, X_val.shape[1], 1)], -1)
        y_pred_reg, y_pred_dist, y_pred_cls = model(inputs, C_val, mask_val)
        
        # Calculate metrics
        y_pred_reg = y_pred_reg.cpu().numpy()[:, 0]
        y_true_reg = y_reg_val.cpu().numpy()
        mape = calculate_mape(y_true_reg, y_pred_reg)
        r2 = calculate_r2(y_true_reg, y_pred_reg)
        mae = calculate_mae(y_true_reg, y_pred_reg)
        
        y_pred_cls = np.argmax(y_pred_cls.cpu().numpy(), -1)
        y_true_cls = y_cls_val.cpu().numpy().astype(np.int64)
        unique_indexes = sorted(np.unique(np.concatenate([y_true_cls, y_pred_cls])).tolist())
        labels2show = [labels[i] for i in unique_indexes]
        res = classification_report(y_true_cls, y_pred_cls, target_names=labels2show)
        acc = accuracy_score(y_true_cls, y_pred_cls)
        
        if not tmp_results is None:
            is_best = r2 >= tmp_results["r2"]
            if is_best:
                tmp_results["mse"] = calculate_mse(y_true_reg, y_pred_reg)
                tmp_results["rmse"] = calculate_rmse(y_true_reg, y_pred_reg)
                tmp_results["mape"] = mape
                tmp_results["mae"] = mae
                tmp_results["r2"] = r2
                
    # print(res)
    res = "\n"+res
    logger.info(f"cls report: {res}")
    logger.info(f"acc: {acc:.4f}")
    logger.info(f"size mae: {mae:.4f}, mape: {mape:.4f}, r2: {r2:.4f}")
    
    if not y_dist_val is None:
        # Calculate metrics
        y_pred_dist = y_pred_dist.cpu().numpy()[:, 0]
        y_true_dist = y_dist_val.cpu().numpy()
        mape = calculate_mape(y_true_dist, y_pred_dist)
        r2 = calculate_r2(y_true_dist, y_pred_dist)
        mae = calculate_mae(y_true_dist, y_pred_dist)
        logger.info(f"dist mae: {mae:.4f}, mape: {mape:.4f}, r2: {r2:.4f}")
        
    return tmp_results, y_true_reg, y_pred_reg, is_best