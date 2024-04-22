import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import argparse
import math
import random
import cv2
import os
from models.model import load_weights, HybridModelV4, HybridModelV5, ModelEMA
from utils.utils import seed_everything
from utils.eval_utils import evaluate_model, evaluate_model_cls
from utils.data_utils import CustomDataset, read_csv_database
from utils.train_utils import init_weights, get_logger, FocalLoss
import pickle as pkl
from sklearn.cluster import KMeans
import pandas as pd

import seaborn as sns
import os
import torch.nn.init as init
from sklearn.metrics import confusion_matrix


def log_reg(fold_eval_results_ema, fold, logger, epoch="None"):
    # fold_eval_results = fold_eval_results_ema
    mape_ema = fold_eval_results_ema["mape"]
    r2_ema = fold_eval_results_ema["r2"]
    mae_ema = fold_eval_results_ema["mae"]

    logger.info(f"- eval - fold: {fold}, epoch: {epoch}, mae (ema): {mae_ema:.4f}")
    logger.info(f"- eval - fold: {fold}, epoch: {epoch}, mape (ema): {mape_ema:.4f}")
    logger.info(f"- eval - fold: {fold}, epoch: {epoch}, r2 (ema): {r2_ema:.4f}")


def log_acc(fold_eval_results_ema_acc, fold, logger, epoch="None"):
    # fold_eval_results = fold_eval_results_ema
    accuracy_ema = fold_eval_results_ema_acc["accuracy"]
    f1_ema = fold_eval_results_ema_acc["f1"]
    recall_ema = fold_eval_results_ema_acc["recall"]
    precision_ema = fold_eval_results_ema_acc["precision"]

    logger.info(
        f"- eval - fold: {fold}, epoch: {epoch}, accuracy (ema): {accuracy_ema:.4f}"
    )
    logger.info(f"- eval - fold: {fold}, epoch: {epoch}, f1 (ema): {f1_ema:.4f}")
    logger.info(
        f"- eval - fold: {fold}, epoch: {epoch}, recall (ema): {recall_ema:.4f}"
    )
    logger.info(
        f"- eval - fold: {fold}, epoch: {epoch}, precision (ema): {precision_ema:.4f}"
    )


def adjust_learning_rate(optimizer, epoch, iter, total_iters):
    if iter < warmup_iters:
        lr = warmup_factor + (initial_lr - warmup_factor) * iter / warmup_iters
    else:
        lr = (
            min_lr
            + (initial_lr - min_lr)
            * (
                1
                + math.cos(
                    math.pi * (iter - warmup_iters) / (total_iters - warmup_iters)
                )
            )
            / 2
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    from utils.flags_set import get_args

    args = get_args()

    with open("data/shapes_list.txt", "r") as f:
        labels = f.read().splitlines()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    exp_dir = f"{args.save_path}/results-{args.exp}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    # encoding='utf-8'

    seed_everything(args.seed)

    # Load the dataset
    conditions = ["Temperature1", "Temperature2", "Time", "Speed"]
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

    if args.focal_loss:
        criterion_cls = FocalLoss(gamma=2)
    else:
        criterion_cls = nn.CrossEntropyLoss()
    criterion_cls2 = nn.BCEWithLogitsLoss()

    if args.load_failure:
        args.num_classes += 1

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

    eval_results = {
        "mse": [],
        "rmse": [],
        "mape": [],
        "mae": [],
        "r2": [],
        "accuracy": [],
        "f1": [],
        "recall": [],
        "precision": [],
    }
    
    y_true_cls_list, y_pred_cls_list = [], []
    for fold in range(5):
        train_dataset = CustomDataset(
            args,
            fold,
            True,
            args.raw_data,
            conditions,
            args.max_threshold,
            scalers_dict,
            logger,
        )
        if args.size_as_input:
            num_conditions = len(conditions) + 1
        else:
            num_conditions = len(conditions)

        tmp_results = {"mse": 100, "rmse": 100, "mape": 100, "mae": 100, "r2": 0.0}
        tmp_results_acc = {"accuracy": 0, "f1": 0, "recall": 0, "precision": 0}
        model = model2use(
            args.model_type,
            args.cls_token,
            args.num_classes,
            args.input_size + num_conditions,
            args.hidden_size,
            args.dim_feedforward,
            args.num_layers,
            args.num_head,
            num_conditions,
            average_feats=args.average_feats,
            with_bn=args.with_bn,
            dropout_rate=args.dropout,
        )
        model.apply(init_weights)

        if args.pretrained:
            logger.info(f"- model - loading from: {args.pretrained}-{fold}-acc.pth")
            load_weights(model, f"{args.pretrained}-{fold}-acc.pth")

        model = model.to(device)
        ema_model = ModelEMA(model, decay=0.99)

        # data to train
        train_data_dict, val_data_dict = {}, {}
        for k, v in train_dataset.data_dict.items():
            train_data_dict[k] = v[train_dataset.train_idx]
            val_data_dict[k] = v[train_dataset.val_idx]

        # val dataset
        X_train = torch.tensor(train_data_dict["X_features"], dtype=torch.float).to(
            device
        )
        C_train = torch.tensor(train_data_dict["C_scaled"], dtype=torch.float).to(
            device
        )
        mask_train = torch.tensor(train_data_dict["mask"], dtype=torch.bool).to(device)
        y_reg_size_train = torch.tensor(
            train_data_dict["y_reg_size"], dtype=torch.float
        ).to(device)
        y_cls_train = torch.tensor(train_data_dict["y_cls"], dtype=torch.float).to(
            device
        )
        logger.info(f"- data - origin train size: {X_train.shape[0]}")

        # val dataset
        X_val = torch.tensor(val_data_dict["X_features"], dtype=torch.float).to(device)
        C_val = torch.tensor(val_data_dict["C_scaled"], dtype=torch.float).to(device)
        mask_val = torch.tensor(val_data_dict["mask"], dtype=torch.bool).to(device)
        y_reg_size_val = torch.tensor(
            val_data_dict["y_reg_size"], dtype=torch.float
        ).to(device)
        y_cls_val = torch.tensor(val_data_dict["y_cls"], dtype=torch.float).to(device)
        logger.info(
            f"- data - train size:{X_train.shape[0]} val size: {X_val.shape[0]}"
        )

        if args.eval_only:
            logger.info("- eval - evaluating val set (ema) ...")
            fold_eval_results_ema_acc, y_true_cls, y_pred_cls, _ = evaluate_model_cls(
                tmp_results_acc,
                model,
                X_val,
                C_val,
                mask_val,
                y_cls_val,
                labels,
                logger,
                args.load_failure,
            )
            log_acc(fold_eval_results_ema_acc, fold, logger)
            
            y_true_cls_list.append(y_true_cls)
            y_pred_cls_list.append(y_pred_cls)

            for metric, value in fold_eval_results_ema_acc.items():
                eval_results[metric].append(value)

        X_val = torch.cat([X_val, X_train])
        C_val = torch.cat([C_val, C_train])
        mask_val = torch.cat([mask_val, mask_train])
        y_cls_val = torch.cat([y_cls_val, y_cls_train])

        model.eval()
        with torch.no_grad():
            inputs = torch.cat(
                [X_val, C_val[:, None, :].repeat(1, X_val.shape[1], 1)], -1
            )
            features = model(inputs, C_val, mask_val, return_cls=True)

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        y_cls_val = y_cls_val.cpu().numpy()
        # Example data loading
        # t-SNE Transformation
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features.cpu().numpy())

        # Plotting
        plt.figure(figsize=(12, 6))
        for i in range(len(labels)):
            plt.scatter(
                features_2d[y_cls_val == i, 0],
                features_2d[y_cls_val == i, 1],
                # label=labels[i],
                label=i,
                edgecolor="k",
                alpha=0.7,
            )
        plt.title("t-SNE visualization of sample features")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.legend(title="Labels")
        plt.savefig(f"{exp_dir}/scatters-{fold}.png")

        plt.close()
        plt.cla()

        # K-means Clustering
        n_clusters = 7  # Set the number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features.cpu().numpy())

        # Plotting
        plt.figure(figsize=(24, 18))
        for i in range(n_clusters):
            plt.scatter(
                features_2d[clusters == i, 0],
                features_2d[clusters == i, 1],
                # label=labels[i],
                label=i,
                edgecolor="k",
                alpha=0.7,
            )

        # Adding labels to each point
        for i, txt in enumerate(y_cls_val):
            plt.annotate(
                int(txt),
                (features_2d[i, 0], features_2d[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Adding a color bar
        plt.title("t-SNE visualization of K-means clustered features")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.savefig(f"{exp_dir}/scatters-kmeans={n_clusters}-{fold}.png")

        plt.close()
        plt.cla()

        data2save = pd.DataFrame({
            "x": features_2d[:, 0].tolist(),
            "y": features_2d[:, 1].tolist(),
            # "shape": [labels[ind] for ind in y_cls_val.astype(np.int64)],
            "shape": y_cls_val.astype(np.int64).tolist(),
            "cluster": clusters.tolist()
        })
        data2save.to_csv(f"{exp_dir}/scatters-kmeans={n_clusters}-{fold}.csv", index=None)

        with open(f"{args.save_path}/{args.exp}.txt", "a") as f:

            accuracy_ema = fold_eval_results_ema_acc["accuracy"]
            f1_ema = fold_eval_results_ema_acc["f1"]
            recall_ema = fold_eval_results_ema_acc["recall"]
            precision_ema = fold_eval_results_ema_acc["precision"]

            f.write(
                f"fold: {fold}, accuracy (ema): {accuracy_ema:.4f}, f1 (ema): {f1_ema:.4f}, recall (ema): {recall_ema:.4f}, precision (ema): {precision_ema:.4f}\n"
            )
            logger.info(f"fold: {fold}, accuracy (ema): {accuracy_ema:.4f}")
            logger.info(f"fold: {fold}, f1 (ema): {f1_ema:.4f}")
            logger.info(f"fold: {fold}, recall (ema): {recall_ema:.4f}")
            logger.info(f"fold: {fold}, precision (ema): {precision_ema:.4f}")

    with open(f"{args.save_path}/{args.exp}.txt", "a") as f:
        f.write("average results:\n")
        for metric in eval_results:
            if len(eval_results[metric]):
                mean_value = np.mean(eval_results[metric])
                logger.info(f"- eval - {metric}: {mean_value:.4f}")
                f.write(f"{metric}: {mean_value:.4f}\n")
        f.write("\n")
        f.write(str(args))
        f.write("\n")
    
    if True:
        y_true_cls = np.concatenate(y_true_cls_list)
        y_pred_cls = np.concatenate(y_pred_cls_list)
        valid_indexes = np.unique(np.concatenate([y_true_cls, y_pred_cls])).tolist()
        valid_labels = [labels[i] for i in valid_indexes]
        # 创建混淆矩阵
        y_true_cls = [labels[i] for i in y_true_cls]
        y_pred_cls = [labels[i] for i in y_pred_cls]
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=valid_labels)

        # 使用seaborn绘图
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=valid_labels, yticklabels=valid_labels)

        # 增加美观性
        plt.title('Confusion Matrix')
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/Confusion-Matrix.png")
        
        # 将混淆矩阵转换为DataFrame，便于保存到CSV
        cm_df = pd.DataFrame(cm, index=valid_labels, columns=valid_labels)

        # 保存到CSV文件
        cm_df.to_csv(f"{exp_dir}/Confusion-Matrix.csv", index=True)
