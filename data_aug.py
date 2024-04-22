import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from utils.utils import seed_everything
from utils.data_utils import CustomDataset
import pickle as pkl
from train import get_logger

if __name__ == "__main__":
    from utils.flags_set import get_args

    # Parse arguments
    args = get_args()
    print(args)

    seed_everything(args.seed)

    # Load the dataset
    conditions = ["Temperature1", "Temperature2", "Time", "Speed"]
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    logger = get_logger(os.path.join(args.save_path, "log.txt"))

    for fold in range(5):

        # 测试集
        train_dataset = CustomDataset(
            args, fold, True, args.raw_data, conditions, args.max_threshold, logger=logger
        )

        with open("data/shapes_list.txt", "r") as f:
            labels = f.read().splitlines()
        indicators = [
            "T",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
        ]
        debug_dict = {
            "ID": [],
            "Temperature1": [],
            "Temperature2": [],
            "Speed": [],
            "Time": [],
            "Num": [],
            "Y1": [],
            "Y2": [],
            "Y3": [],
        }
        for v in indicators:
            debug_dict[f"{v}1"] = []
            debug_dict[f"{v}2"] = []
        for idx in tqdm(range(len(train_dataset))):
            for r in range(10):
                (
                    x_id,
                    x_objects,
                    x_mol,
                    x_con,
                    x_mask,
                    x_num,
                    y_reg_size,
                    y_reg_dist,
                    y_cls,
                    sucess,
                ) = train_dataset.debug(idx)
                debug_dict["ID"].append(x_id)
                debug_dict["Temperature1"].append(x_con[0])
                debug_dict["Temperature2"].append(x_con[1])
                debug_dict["Time"].append(x_con[2])
                debug_dict["Speed"].append(x_con[3])
                debug_dict["Num"].append(x_num)
                debug_dict["Y1"].append(y_reg_size)
                debug_dict["Y2"].append(y_reg_dist)
                debug_dict["Y3"].append(y_cls)
                for i, v in enumerate(indicators):
                    debug_dict[f"{v}1"].append(x_mol[i])
                    debug_dict[f"{v}2"].append(x_objects[i])
        pd.DataFrame(debug_dict).to_csv(
            f"{args.save_path}/fold{fold}-{args.max_threshold}.csv",
            index=None,
        )

    with open(f"{args.save_path}/scalers-{args.max_threshold}.pkl", "wb") as f:
        pkl.dump(train_dataset.scalers_dict, f)
