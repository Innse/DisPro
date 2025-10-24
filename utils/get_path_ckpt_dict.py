#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_path_ckpt_dict.py
@Time    :   2024/10/08 20:06:05
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib

import os
import json
import pandas as pd


result_root = "/path/to/results/DisPro/"
modal = "Omics"
# modal = "WSI"
save_json = f"splits/ckpts_unipro_{modal}.json"

studies = ["BLCA", "BRCA", "COADREAD", "LUAD", "UCEC"]


rates_wsi = [0, 0, 20, 30, 40, 60]
rates_omics = [0, 60, 40, 30, 20, 0]
# rates_wsi = [40]
# rates_omics = [20]

if modal == 'Omics':
    model_name = "Coop_PathTrans_BioBert"
else:
    model_name = "Coop_WSI_BioBert"

path_ckpt_dict = {}
for study in studies:
    for r_wsi, r_omics in zip(rates_wsi, rates_omics):
        dir_ckpt = os.path.join(result_root, modal, study)
        miss_setup = f"W{r_wsi}_O{r_omics}"
        prefix = f"[{model_name}]-[{miss_setup}]"
        # find latest ckpt
        result_csv = None
        if not os.path.exists(dir_ckpt):
            print(f"{dir_ckpt} not exist.")
            continue
        for ckpt in sorted(os.listdir(dir_ckpt), reverse=True):
            if ckpt.startswith(prefix):
                for file in os.listdir(os.path.join(dir_ckpt, ckpt)):
                    if file.startswith('results_') and file.endswith('.csv'):
                        result_csv = os.path.join(dir_ckpt, ckpt, file)
            if result_csv:
                df = pd.read_csv(result_csv, index_col=0)
                # print mean and std
                print(f"{study}: {ckpt}")
                print(f"{study}: {ckpt} \nmean: {df['mean']}, std: {df['std']}")
                    
                # get ckpt date and time
                date_time = ckpt.split("-")[2:]
                date_time = "-".join(date_time)
                print(date_time)
                if study not in path_ckpt_dict.keys():
                    path_ckpt_dict[f"{study}"] = {}
                path_ckpt_dict[f"{study}"][miss_setup] = date_time
                break
        if result_csv is None:
            print(f"Cannot find path for ckpts of {prefix}")


print(path_ckpt_dict)
with open(save_json, 'w') as f:
    json.dump(path_ckpt_dict, f, indent=4, sort_keys=True)