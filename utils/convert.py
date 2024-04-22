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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.class_weight import compute_class_weight
import torch_geometric.nn as geom_nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import math
import cv2
import os

def aug_raw_data(file_path):
    data = pd.read_csv(file_path)
    data = data[[
        'ID', 'Temperature1', 'Temperature2', 'Speed', 'Time', 'Num', 
        'Y1', 'Y2', 'Y3', 'T1', 'T2', 
        'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'J1',
        'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2', 'J2'
    ]]
    mean_value = data['Speed'][data['Speed'].fillna(value=-1) > 0].mean()
    print("mean_value Speed:", mean_value)
    data['Speed'].fillna(value=mean_value, inplace=True)
    data['Num'].fillna(value=-1, inplace=True)
    mean_value = data['Y2'][data['Y2'].fillna(value=-1) > 0].mean()
    data['Y2'].fillna(value=mean_value, inplace=True)
    print("mean_value Y2:", mean_value)
    data = data.dropna(axis=0, how='any')
    shapes_list = np.unique(data['Y3']).tolist()
    name_to_index = {name: index for index, name in enumerate(shapes_list)}
    data['Y3'] = data['Y3'].map(name_to_index)

    with open("data/shapes_list.txt", "w") as f:
        for lbl in shapes_list:
            f.write(lbl+"\n")
    N = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    X1 = data[[v+"1" for v in N]]
    X2 = data[[v+"2" for v in N]]
    extra_data = []
    
    for idx in tqdm(range(X2.shape[0])):
        Volumes = [v for v in X1.iloc[idx]]
        Names = [v for v in X2.iloc[idx]]
        for name, volume, ndx in zip(Names, Volumes, N):
            if name in intern_dict:
                for name_2, volume_2, ndx_2 in zip(Names, Volumes, N):
                    if not name_2 == name and name_2 in intern_dict[name]:
                        indicator = f'{name}@{name_2}'
                        assert os.path.exists(os.path.join("../Quantum-ML0109/all_quantums_1214/intern", indicator+".cif"))
                        tmp_data = {
                            "ID": data.iloc[idx]["ID"],
                            "Temperature1": data.iloc[idx]["Temperature1"],
                            "Temperature2": data.iloc[idx]["Temperature2"],
                            "Speed": data.iloc[idx]["Speed"],
                            "Time": data.iloc[idx]["Time"],
                            "Num": data.iloc[idx]["Num"],
                            "Y1": data.iloc[idx]["Y1"],
                            "T1": data.iloc[idx]["T1"],
                            "T2": data.iloc[idx]["T2"]
                        }
                        for n in N:
                            tmp_data[f"{n}1"] = X1.iloc[idx][f"{n}1"]
                        for n in N:
                            tmp_data[f"{n}2"] = X2.iloc[idx][f"{n}2"]
                            
                        if volume > volume_2:
                            tmp_data[f"{ndx}1"] -= volume_2
                            tmp_data[f"{ndx_2}1"] = volume_2
                            tmp_data[f"{ndx_2}2"] = indicator
                        elif volume < volume_2:
                            tmp_data[f"{ndx_2}1"] -= volume
                            tmp_data[f"{ndx}1"] = volume
                            tmp_data[f"{ndx}2"] = indicator
                        else: # 相等
                            tmp_data[f"{ndx}1"] = volume
                            tmp_data[f"{ndx}2"] = indicator
                            tmp_data[f"{ndx_2}1"] = 0.0
                            tmp_data[f"{ndx_2}2"] = "PlaceHolder"
                        extra_data.append(tmp_data)
    
    print("total:", len(extra_data), "/", len(data))
    extra_data = pd.DataFrame(extra_data)
    return data, extra_data

if __name__ == '__main__':
    
    intern_dict = {}
    for cif in os.listdir("../Quantum-ML0109/all_quantums_1214/intern"):
        name = cif.split(".")[0]
        a1, a2 = name.split("@")
        if not a1 in intern_dict:
            intern_dict[a1] = []
        if not a2 in intern_dict[a1]:
            intern_dict[a1].append(a2)
            
    # Load the dataset
    file_path = 'data/raw_selected0128_classification_and_reg.csv'

    data, extra_data = aug_raw_data(file_path)
    extra_data.to_csv("data/0201_classification_and_reg-extra-v0.csv", index=None)
    data.to_csv("data/0201_classification_and_reg-raw-v2.csv", index=None)
    
    # for idx in range(5):
    #     # Load the dataset
    #     file_path = f'data/0128_classification_and_reg-extra-v{idx}.csv'

    #     data, extra_data = aug_raw_data(file_path)
    #     extra_data = pd.concat([data, extra_data])
    #     idx_ = idx + 1
    #     extra_data.to_csv(f"data/0128_classification_and_reg-extra-v{idx_}.csv", index=None)
