import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import Sampler
# import wandb


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter():
    def __init__(self, fold=5):
        self.fold = fold
        self.header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
        self.epochs = ["epoch"]
        self.cindex = ["cindex"]
        self.results = {}

    def updata(self, results):
        if not isinstance(results, dict):
            epoch, score = results
            self.epochs.append(epoch)
            self.cindex.append(round(score, 4))
        else:
            '''
            results = {
            "wsi": {"c-index": 0, "checkpoint": None, "epoch": 0},
            "omics": {"c-index": 0, "checkpoint": None, "epoch": 0},
            "wsi-omics": {"c-index": 0, "checkpoint": None, "epoch": 0},
            }
            '''
            for modals, v in results.items():
                self.results.setdefault(modals, {})
                self.results[modals].setdefault("c-index", [])
                self.results[modals].setdefault("epoch", [])
                self.results[modals]["c-index"].append(v["c-index"])
                self.results[modals]["epoch"].append(v["epoch"])

    def save(self, path):
        if len(self.results) == 0:
            mean = round(np.mean(self.cindex[1:self.fold + 1]), 4)
            std = round(np.std(self.cindex[1:self.fold + 1]), 4)
            self.cindex.append(mean)
            self.cindex.append(std)
            # wandb.log({"mean_cindex": mean, "std_cindex": std})
            print("save evaluation resluts to", path)
            with open(path, "w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(self.header)
                writer.writerow(self.epochs)
                writer.writerow(self.cindex)
        else:
            for modals, v in self.results.items():
                mean = round(np.mean(v["c-index"]), 4)
                std = round(np.std(v["c-index"]), 4)
                v["c-index"].append(mean)
                v["c-index"].append(std)
                # wandb.log({"mean_{}_cindex".format(modals): mean, "std_{}_cindex".format(modals): std})
            print("save evaluation resluts to", path)
            with open(path, "w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(self.header)
                for modals, v in self.results.items():
                    writer.writerow([modals] + v["c-index"])
                    writer.writerow([modals+"_epoch"] + v["epoch"])
                    
def loading_unipro_config():
    import argparse
    import json
    missing_modal_config = argparse.Namespace()
    with open("splits/ckpts_unipro_WSI.json", 'r') as f:
        ckpt_dict_wsi = json.load(f)
    with open("splits/ckpts_unipro_Omics.json", 'r') as f:
        ckpt_dict_omics = json.load(f)

    missing_modal_config.path_model_name = "Coop_WSI_BioBert"
    missing_modal_config.path_ckpt_dict = ckpt_dict_wsi
    
    missing_modal_config.omics_model_name = "Coop_PathTrans_BioBert"
    missing_modal_config.omics_ckpt_dict = ckpt_dict_omics

    
    return missing_modal_config


def load_uni_models_for_missing(fold, missing_modal_config, args):
    study = args.study
    name_model_wsi = "[{wsi_model_name}]-[{miss_suffix}]-{timestamp}".format(
            wsi_model_name=missing_modal_config.path_model_name,
            miss_suffix=args.miss_suffix,
            timestamp=missing_modal_config.path_ckpt_dict[study][args.miss_suffix])
    
    dir_model_wsi = os.path.join(
        args.result_dir,
        "WSI",
        study,
        name_model_wsi,
        "fold_{}".format(fold),
    )
    if not os.path.exists(dir_model_wsi):
        raise ValueError(
                "Model directory not found: {}".format(dir_model_wsi))
    else:
        # find the checkpoint with ".pth.tar" extension
        list_files = os.listdir(dir_model_wsi)
        list_files = [f for f in list_files if f.endswith(".pth.tar")]
        if len(list_files) == 0:
            raise ValueError(
                "No checkpoint found in the directory: {}".format(dir_model_wsi))
        elif len(list_files) > 1:
            raise ValueError(
                "Multiple checkpoints found in the directory: {}".format(dir_model_wsi))
        else:
            path_model_wsi = os.path.join(dir_model_wsi, list_files[0])
            print("WSI Model path: {}".format(path_model_wsi))
    
    # ---> Omics model
    name_model_omics = "[{omics_model_name}]-[{miss_suffix}]-{timestamp}".format(
            omics_model_name=missing_modal_config.omics_model_name,
            miss_suffix=args.miss_suffix,
            timestamp=missing_modal_config.omics_ckpt_dict[study][args.miss_suffix])
    
    dir_model_omics = os.path.join(
        args.result_dir,
        "Omics",
        study,
        name_model_omics,
        "fold_{}".format(fold),
    )
    if not os.path.exists(dir_model_omics):
        raise ValueError(
                "Model directory not found: {}".format(dir_model_omics))
    else:
        # find the checkpoint with ".pth.tar" extension
        list_files = os.listdir(dir_model_omics)
        list_files = [f for f in list_files if f.endswith(".pth.tar")]
        if len(list_files) == 0:
            raise ValueError(
                "No checkpoint found in the directory: {}".format(dir_model_omics))
        elif len(list_files) > 1:
            raise ValueError(
                "Multiple checkpoints found in the directory: {}".format(dir_model_omics))
        else:
            path_model_omics = os.path.join(dir_model_omics, list_files[0])
            print("Omics Model path: {}".format(path_model_omics))
    return path_model_wsi, path_model_omics