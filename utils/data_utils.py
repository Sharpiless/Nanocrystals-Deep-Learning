import numpy as np
import pandas as pd
import pickle
import random
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import torch
from utils.scalers import sklearn_scalers_dict


def read_csv_database(
    args,
    file_path,
    conditions,
    max_threshold=200,
    substance_features=None,
    scalers_dict=None,
    indicators=["T", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
    logger=None,
    fail_data=False
):
    if args.without_product:
        assert not args.mask_target
        
    logger.info("- data - loading raw csv data ...")
    csv_data = pd.read_csv(file_path)  # 加载原始 csv 数据
    logger.info(f"   - before process: {csv_data.shape}")
    csv_data = csv_data[csv_data["Y1"] <= max_threshold]  # 按照尺寸最大阈值进行过滤
    logger.info(f"   - after filter threshold ({max_threshold}): {csv_data.shape}")

    X1 = csv_data[[f"{v}1" for v in indicators]]  # Mol 量
    X2 = csv_data[[f"{v}2" for v in indicators]]  # 生成物+反应物
    X3 = csv_data[conditions]  # 反应条件
    if args.without_product:
        X1 = X1.drop('T1', axis=1) 
        X2 = X2.drop('T2', axis=1) 

    if scalers_dict:  # 如果有预先拟合的 scalers
        logger.info("- scaler - use pre-defined scalers")
        # Standardize 'Mol'
        mol_scaler = scalers_dict["mol_scaler"]
        T_scaler = scalers_dict["T_scaler"]
        C_scaler = scalers_dict["C_scaler"]

        # Standardize 'Mol'
        X1_scaled = mol_scaler.transform(np.array(X1).flatten()[:, None]).reshape(
            X1.shape
        )[
            :, :, None
        ]  # 标准化后的mol量
        # Standardize 'Conditions'
        T_scaled = T_scaler.transform(
            X3[["Temperature1", "Temperature2"]]
        )  # 标准化后的反应温度
        C_scaled = np.array(
            C_scaler.transform(X3[["Time", "Speed"]])
        )  # 标准化后的反应条件
        
        if args.size_as_input:
            if fail_data:
                sizes = np.random.normal(loc=0, scale=1, size=csv_data[["Y1"]].values.shape)
                sizes = np.abs(sizes) * 20
            else:
                sizes = csv_data[["Y1"]]
            S_scaler = scalers_dict["S_scaler"]
            S_scaled = S_scaler.transform(
                sizes
            )  # 标准化后的尺寸
            C_scaled = np.concatenate([C_scaled, S_scaled], -1)
            scalers_dict["S_scaler"] = S_scaler
    else:
        logger.info("- scaler - use new scalers")
        scalers_dict = {}
        # Standardize
        mol_scaler = sklearn_scalers_dict[args.scaler_m]()
        T_scaler = sklearn_scalers_dict[args.scaler_t]()
        C_scaler = sklearn_scalers_dict[args.scaler_m]()

        # Standardize 'Mol'
        X1_scaled = mol_scaler.fit_transform(np.array(X1).flatten()[:, None]).reshape(
            X1.shape
        )[
            :, :, None
        ]  # 标准化后的mol量
        # Standardize 'Conditions'
        T_scaled = T_scaler.fit_transform(
            X3[["Temperature1", "Temperature2"]]
        )  # 标准化后的反应温度
        C_scaled = np.array(
            C_scaler.fit_transform(X3[["Time", "Speed"]])
        )  # 标准化后的反应条件
        scalers_dict["mol_scaler"] = mol_scaler
        scalers_dict["T_scaler"] = T_scaler
        scalers_dict["C_scaler"] = C_scaler
        
        if args.size_as_input:
            S_scaler = sklearn_scalers_dict[args.scaler_t]()
            if fail_data:
                sizes = np.random.normal(loc=0, scale=1, size=csv_data[["Y1"]].values.shape)
                sizes = np.abs(sizes) * 20
            else:
                sizes = csv_data[["Y1"]]
            S_scaler = scalers_dict["S_scaler"]
            S_scaled = S_scaler.fit_transform(
                sizes
            )  # 标准化后的尺寸
            C_scaled = np.concatenate([C_scaled, S_scaled], -1)
            scalers_dict["S_scaler"] = S_scaler

    C_scaled = np.concatenate([T_scaled, C_scaled], -1)
    if args.save_scalers:
        with open(args.save_scalers, "wb") as f:
            pickle.dump(scalers_dict, f)

    C_unscaled = X3.values  # 反应条件原始数据
    X_mol = X1.values
    # Create a mask where 'X1' is zero
    mask = (X1 == 0).values  # 占位符

    # Load names
    X2_list = []
    for col in X2.columns:
        tmp = []
        for substance in X2[col]:
            if substance in substance_features:
                tmp.append(substance)
            else:
                logger.info(f"- error - {substance} not found in .pkl file")
                tmp.append("PlaceHolder")
        X2_list.append(tmp)

    # Load features
    X2_features = []
    for col in X2.columns:
        features = []
        for substance in X2[col]:
            if substance in substance_features:
                features.append(substance_features[substance][None, :])
            else:
                features.append(substance_features["PlaceHolder"][None, :])
        X2_features.append(np.array(features))
    X2_features = np.concatenate(X2_features, axis=1)
        
    # Concatenate indicator and mol
    indicators = np.zeros((X2_features.shape[0], X2_features.shape[1], 1))
    if not args.without_product:
        indicators[:, :1] = 1.0  # 目标反应物
    X2_features = np.concatenate([indicators, X1_scaled, X2_features], axis=-1)

    objects_list = []
    for i in range(len(X2_list[0])):
        objects = []
        for j in range(len(X2_list)):
            objects.append(X2_list[j][i])
        objects_list.append(objects)

    # Splitting labels for classification# Load reg labels
    y_reg_size = csv_data["Y1"].values
    y_reg_dist = csv_data["Y2"].values
    y_cls = csv_data["Y3"].values
    ids = csv_data["ID"].values
    nums = csv_data["Num"].values
    indexes = np.arange(nums.shape[0])
        
    data_dict = {
        "X_objects": np.array(objects_list),
        "X_products": csv_data["T2"].values,
        "X_features": X2_features,
        "C_scaled": C_scaled,
        "C_unscaled": C_unscaled,
        "X_mol": X_mol,
        "mask": mask,
        "y_reg_size": y_reg_size,
        "y_reg_dist": y_reg_dist,
        "y_cls": y_cls,
        "ids": ids,
        "nums": nums,
    }
    return data_dict, scalers_dict, csv_data

class CustomDatasetALL(object):
    def __init__(
        self,
        args,
        fold,
        training,
        file_path,
        conditions,
        max_threshold,
        scalers_dict=None,
        logger=None,
    ):
        self.args = args
        self.fold = fold
        self.training = training
        self.conditions = conditions
        self.scalers_dict = scalers_dict
        self.logger = logger
        # Load the .pkl file
        with open(args.feats, "rb") as f:
            self.obj_features = pickle.load(f)
        self.logger.info(f"- model - load feat file from: {args.feats}")
        self.preprocess_data()
        if self.args.expand_data:
            self.expand_dict = {}
            for name in self.obj_features:
                if "@" in name:
                    A1, A2 = name.split("@")
                    if not A1 in self.expand_dict:
                        self.expand_dict[A1] = []
                    self.expand_dict[A1].append(A2)
            self.logger.info(" - data - total intern data: {len(self.expand_dict)}")

    def preprocess_data(self):

        args = self.args

        self.data_dict, self.scalers_dict, self.csv_data = read_csv_database(
            self.args,
            self.args.raw_data,
            self.conditions,
            self.args.max_threshold,
            self.obj_features,
            self.scalers_dict,
            logger=self.logger,
        )
        self.mol_scaler = self.scalers_dict["mol_scaler"]
        # 可用五折数据的所有indexes（初始化）
        self.train_idx = np.arange(self.data_dict["nums"].shape[0])
        
        self.logger.info(f" - data - update train_num: {self.train_idx.shape[0]}")

    def __len__(self):
        if self.training:
            return len(self.train_idx)
        else:
            return len(self.val_idx)

    def expand_data(self, index, x_objects, x_mask, x_mol):

        if True:
            random_indexes = np.arange(x_mol.shape[0])[1:]
            random.shuffle(random_indexes)
            sucess = False
            for i in random_indexes:
                a1, m1 = x_objects[i], x_mol[i]
                if a1 in self.expand_dict and m1 > 0:
                    for j in random_indexes:
                        a2, m2 = x_objects[j], x_mol[j]
                        if a2 in self.expand_dict[a1] and m2 > 0:
                            name = f"{a1}@{a2}"
                            sucess = True
                            break
                if sucess:
                    break
            if not sucess:
                return x_objects, x_mask, x_mol, sucess
            if m1 > m2:
                x_mol[i] = x_mol[i] - x_mol[j]  # a2 被反应完了
                x_objects[j] = name
            elif m2 > m1:
                x_mol[j] = x_mol[j] - x_mol[i]  # a1 被反应完了
                x_objects[i] = name
            elif m2 == m1:
                x_mol[j] = 0  # 都被反应完了
                x_objects[i] = name
                x_objects[j] = "PlaceHolder"
                x_mask[j] = True
        if self.args.debug:
            self.logger.info(
                "v1:", self.args.expand_data, sucess, a1, m1, a2, m2, x_objects
            )
        return x_objects, x_mask, x_mol, sucess

    def expand_data_v2(self, index, x_objects, x_mask, x_mol):
        sucess = False
        if "PlaceHolder" in x_objects[:11]:
            random_indexes = np.arange(x_mol.shape[0])[1:]
            random.shuffle(random_indexes)
            for i in random_indexes:
                a1, m1 = x_objects[i], x_mol[i]
                if a1 in self.expand_dict and m1 > 0:
                    for j in random_indexes:
                        a2, m2 = x_objects[j], x_mol[j]
                        if a2 in self.expand_dict[a1] and m2 > 0:
                            name = f"{a1}@{a2}"
                            sucess = True
                            break
                if sucess:
                    break
            if not sucess:
                return x_objects, x_mask, x_mol, sucess

            if np.random.random() > 0.5:
                rand_mol = np.random.random() * min(m1, m2)
            else:
                rand_mol = min(m1, m2)
            x_mol[j] = x_mol[j] - rand_mol  # a2 反应
            x_mol[i] = x_mol[i] - rand_mol  # a1 反应

            if name in x_objects:
                placeholder_idx = x_objects.index(name)
            else:
                placeholder_idx = x_objects.index("PlaceHolder")
            x_objects[placeholder_idx] = name  # 把生成物放到填充位置
            x_mol[placeholder_idx] += rand_mol
            x_mask[placeholder_idx] = False
            
        return x_objects, x_mask, x_mol, sucess

    def debug(self, idx):
        index = self.train_idx[idx]

        x_con_unscaled = self.data_dict["C_unscaled"][index].copy()
        x_num = self.data_dict["nums"][index].copy()
        x_id = self.data_dict["ids"][index].copy()
        indicators = np.zeros((self.data_dict["X_objects"][index].shape[0], 1))
        indicators[0, 0] = 1

        # Load data from data_paths using idx
        x_objects = self.data_dict["X_objects"][index].copy().tolist()
        x_mask = self.data_dict["mask"][index].copy()
        x_mol = self.data_dict["X_mol"][index].copy()

        sucess = True
        for r in range(random.randint(1, 3)):
            if self.args.debug:
                print("repeat:", r)
            if np.random.random() > 0.5:
                x_objects, x_mask, x_mol, r_sucess = self.expand_data_v2(
                    index, x_objects, x_mask, x_mol
                )
            else:
                x_objects, x_mask, x_mol, r_sucess = self.expand_data(
                    index, x_objects, x_mask, x_mol
                )
            if not r_sucess and r == 0:
                sucess = False
                break

        # load labels
        y_reg_size = self.data_dict["y_reg_size"][index].copy()
        y_reg_dist = self.data_dict["y_reg_dist"][index].copy()
        y_cls = self.data_dict["y_cls"][index].copy()
        return (
            x_id,
            x_objects,
            x_mol,
            x_con_unscaled,
            x_mask,
            x_num,
            y_reg_size,
            y_reg_dist,
            y_cls,
            sucess,
        )

class CustomDataset(object):
    def __init__(
        self,
        args,
        fold,
        training,
        file_path,
        conditions,
        max_threshold,
        scalers_dict=None,
        logger=None,
    ):
        self.args = args
        self.fold = fold
        self.training = training
        self.conditions = conditions
        self.scalers_dict = scalers_dict
        self.logger = logger
        # Load the .pkl file
        with open(args.feats, "rb") as f:
            self.obj_features = pickle.load(f)
        self.logger.info(f"- model - load feat file from: {args.feats}")
        self.kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        self.preprocess_data()
        if self.args.expand_data:
            self.expand_dict = {}
            for name in self.obj_features:
                if "@" in name:
                    A1, A2 = name.split("@")
                    if not A1 in self.expand_dict:
                        self.expand_dict[A1] = []
                    self.expand_dict[A1].append(A2)
            self.logger.info(" - data - total intern data: {len(self.expand_dict)}")

    def preprocess_data(self):

        args = self.args

        self.data_dict, self.scalers_dict, self.csv_data = read_csv_database(
            self.args,
            self.args.raw_data,
            self.conditions,
            self.args.max_threshold,
            self.obj_features,
            self.scalers_dict,
            logger=self.logger,
        )
        self.mol_scaler = self.scalers_dict["mol_scaler"]
        # 可用五折数据的所有indexes（初始化）
        self.data_indexes = np.arange(self.data_dict["nums"].shape[0])
        # 删去统计数量少的数据索引
        self.num_thre = self.args.num_threshold
        if self.args.weak_data:
            self.weak_data_indexes = self.data_indexes[
                self.data_dict["nums"] < self.num_thre
            ]
        else:
            self.weak_data_indexes = None

        # 保留统计数量多的数据索引
        self.data_indexes = self.data_indexes[self.data_dict["nums"] >= self.num_thre]
        self.logger.info(
            f"""- data - num thre: {self.num_thre}, \
                solid data: {self.data_indexes.shape[0]}, \
                weak data: {self.weak_data_indexes.shape[0]}"""
        )

        # extra data 大于测试集标签最大值、小于数据裁剪最大值
        if args.max_threshold > args.threshold:
            intern_size_mask = (
                self.data_dict["y_reg_size"][self.data_indexes] > args.threshold
            )
            self.ext_data_indexes = self.data_indexes[intern_size_mask]
            self.data_indexes = self.data_indexes[np.logical_not(intern_size_mask)]

            self.logger.info(
                f"- data - filter thre={args.threshold}-{args.max_threshold}, keep data: {self.data_indexes.shape}"
            )
            self.logger.info(
                f"- data - filter thre={args.threshold}-{args.max_threshold}, extra data: {self.ext_data_indexes.shape}"
            )
        else:
            self.ext_data_indexes = None

        all_folds = [
            (train_idx, val_idx)
            for (train_idx, val_idx) in self.kf.split(self.data_indexes)
        ]
        self.train_idx, self.val_idx = all_folds[self.fold]

        self.logger.info(
            f"- data - train_num: {self.train_idx.shape[0]}, val num: {self.val_idx.shape[0]}"
        )
        self.val_idx = self.data_indexes[self.val_idx]

        # concat training data and index
        if not self.ext_data_indexes is None:
            self.train_idx = np.concatenate(
                [
                    self.data_indexes[self.train_idx],
                    self.weak_data_indexes,
                    self.ext_data_indexes
                ]
            )
        else:
            self.train_idx = np.concatenate(
                [
                    self.data_indexes[self.train_idx],
                    self.weak_data_indexes
                ]
            )
        self.logger.info(f" - data - update train_num: {self.train_idx.shape[0]}")

    def __len__(self):
        if self.training:
            return len(self.train_idx)
        else:
            return len(self.val_idx)

    def expand_data(self, index, x_objects, x_mask, x_mol):

        if True:
            random_indexes = np.arange(x_mol.shape[0])[1:]
            random.shuffle(random_indexes)
            sucess = False
            for i in random_indexes:
                a1, m1 = x_objects[i], x_mol[i]
                if a1 in self.expand_dict and m1 > 0:
                    for j in random_indexes:
                        a2, m2 = x_objects[j], x_mol[j]
                        if a2 in self.expand_dict[a1] and m2 > 0:
                            name = f"{a1}@{a2}"
                            sucess = True
                            break
                if sucess:
                    break
            if not sucess:
                return x_objects, x_mask, x_mol, sucess
            if m1 > m2:
                x_mol[i] = x_mol[i] - x_mol[j]  # a2 被反应完了
                x_objects[j] = name
            elif m2 > m1:
                x_mol[j] = x_mol[j] - x_mol[i]  # a1 被反应完了
                x_objects[i] = name
            elif m2 == m1:
                x_mol[j] = 0  # 都被反应完了
                x_objects[i] = name
                x_objects[j] = "PlaceHolder"
                x_mask[j] = True
        if self.args.debug:
            self.logger.info(
                "v1:", self.args.expand_data, sucess, a1, m1, a2, m2, x_objects
            )
        return x_objects, x_mask, x_mol, sucess

    def expand_data_v2(self, index, x_objects, x_mask, x_mol):
        sucess = False
        if "PlaceHolder" in x_objects[:11]:
            random_indexes = np.arange(x_mol.shape[0])[1:]
            random.shuffle(random_indexes)
            for i in random_indexes:
                a1, m1 = x_objects[i], x_mol[i]
                if a1 in self.expand_dict and m1 > 0:
                    for j in random_indexes:
                        a2, m2 = x_objects[j], x_mol[j]
                        if a2 in self.expand_dict[a1] and m2 > 0:
                            name = f"{a1}@{a2}"
                            sucess = True
                            break
                if sucess:
                    break
            if not sucess:
                return x_objects, x_mask, x_mol, sucess

            if np.random.random() > 0.5:
                rand_mol = np.random.random() * min(m1, m2)
            else:
                rand_mol = min(m1, m2)
            x_mol[j] = x_mol[j] - rand_mol  # a2 反应
            x_mol[i] = x_mol[i] - rand_mol  # a1 反应

            if name in x_objects:
                placeholder_idx = x_objects.index(name)
            else:
                placeholder_idx = x_objects.index("PlaceHolder")
            x_objects[placeholder_idx] = name  # 把生成物放到填充位置
            x_mol[placeholder_idx] += rand_mol
            x_mask[placeholder_idx] = False
            
        return x_objects, x_mask, x_mol, sucess

    def debug(self, idx):
        index = self.train_idx[idx]

        x_con_unscaled = self.data_dict["C_unscaled"][index].copy()
        x_num = self.data_dict["nums"][index].copy()
        x_id = self.data_dict["ids"][index].copy()
        indicators = np.zeros((self.data_dict["X_objects"][index].shape[0], 1))
        indicators[0, 0] = 1

        # Load data from data_paths using idx
        x_objects = self.data_dict["X_objects"][index].copy().tolist()
        x_mask = self.data_dict["mask"][index].copy()
        x_mol = self.data_dict["X_mol"][index].copy()

        sucess = True
        for r in range(random.randint(1, 3)):
            if self.args.debug:
                print("repeat:", r)
            if np.random.random() > 0.5:
                x_objects, x_mask, x_mol, r_sucess = self.expand_data_v2(
                    index, x_objects, x_mask, x_mol
                )
            else:
                x_objects, x_mask, x_mol, r_sucess = self.expand_data(
                    index, x_objects, x_mask, x_mol
                )
            if not r_sucess and r == 0:
                sucess = False
                break

        # load labels
        y_reg_size = self.data_dict["y_reg_size"][index].copy()
        y_reg_dist = self.data_dict["y_reg_dist"][index].copy()
        y_cls = self.data_dict["y_cls"][index].copy()
        return (
            x_id,
            x_objects,
            x_mol,
            x_con_unscaled,
            x_mask,
            x_num,
            y_reg_size,
            y_reg_dist,
            y_cls,
            sucess,
        )

class ObjectDataset(object):
    def __init__(
        self,
        args,
        object_name,
        training,
        file_path,
        conditions,
        max_threshold,
        scalers_dict=None,
        logger=None,
    ):
        self.args = args
        self.object_name = object_name
        self.training = training
        self.conditions = conditions
        self.scalers_dict = scalers_dict
        self.logger = logger
        # Load the .pkl file
        with open(args.feats, "rb") as f:
            self.obj_features = pickle.load(f)
        self.logger.info(f"- model - load feat file from: {args.feats}")
        self.preprocess_data()
        if self.args.expand_data:
            self.expand_dict = {}
            for name in self.obj_features:
                if "@" in name:
                    A1, A2 = name.split("@")
                    if not A1 in self.expand_dict:
                        self.expand_dict[A1] = []
                    self.expand_dict[A1].append(A2)
            self.logger.info(" - data - total intern data: {len(self.expand_dict)}")

    def preprocess_data(self):

        args = self.args

        self.data_dict, self.scalers_dict, self.csv_data = read_csv_database(
            self.args,
            self.args.raw_data,
            self.conditions,
            self.args.max_threshold,
            self.obj_features,
            self.scalers_dict,
            logger=self.logger,
        )
        self.mol_scaler = self.scalers_dict["mol_scaler"]
        # 可用数据的所有indexes（初始化）
        self.data_indexes = np.arange(self.data_dict["nums"].shape[0])
        
        self.val_idx = self.data_indexes[
            np.logical_and(
                self.data_dict['X_products'] == self.object_name,
                self.data_dict["y_reg_size"] <= args.threshold
            )
        ]
        
        self.train_idx = self.data_indexes[np.logical_not(
                self.data_dict['X_products'] == self.object_name
            )]

        self.logger.info(
            f"- data - train_num: {self.train_idx.shape[0]}, val num: {self.val_idx.shape[0]}"
        )


    def __len__(self):
        if self.training:
            return len(self.train_idx)
        else:
            return len(self.val_idx)
