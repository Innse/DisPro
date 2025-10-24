import imp
import os
from random import shuffle
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.utils.data as data


class TCGA_Dataset(data.Dataset):
    def __init__(self, excel_file, modal, signatures=None, data_root_wsi=None, data_root_omics=None):
        self.modal = modal
        self.signatures = signatures
        self.data_root_wsi = data_root_wsi
        self.data_root_omics = data_root_omics

        print("[dataset] loading dataset from %s" % (excel_file))
        assert os.path.exists(excel_file), "file [{}] not found".format(excel_file)
        self.rows = pd.read_csv(excel_file)

        if "Label" not in self.rows.columns:
            self.rows = self.disc_label(self.rows)
        #
        if self.signatures:
            assert os.path.exists(signatures), "file [{}] not found".format(signatures)
            print("[dataset] loading signatures from %s" % (signatures))
            df = pd.read_csv(signatures)
            self.signatures = []
            for col in df.columns:
                signature = df[col].dropna().unique()
                self.signatures.append(signature)
        #
        if "Omics" == self.modal:
            # 去掉"Omics"列为空值的行
            self.rows = self.rows.dropna(subset=["Omics"]).reset_index(drop=True)
            # 取["Omics"]列的第一个非空值来读取数据以获取维度信息
            omics_example = self.rows["Omics"].dropna().values.tolist()[0]
            self.omics_size, _ = self.read_Omics(omics_example)
            self.omics_size = [len(omics) for omics in self.omics_size]
            print(f"[dataset] sizes of omics data: ({len(self.omics_size)}) {self.omics_size}")
            

        if "WSI" == self.modal:
            # 去掉"WSI"列为空值的行
            self.rows = self.rows.dropna(subset=["WSI"]).reset_index(drop=True)
            # 取["WSI"]列的第一个非空值来读取数据以获取维度信息
            wsi_example = self.rows["WSI"].dropna().values.tolist()[0]
            self.path_size, _ = self.read_WSI(wsi_example)
            self.path_size = self.path_size.size(-1)
            print(f"[dataset] dimension of WSI features: {self.path_size}")
        
        
        self.splits = {split: [i for i, x in enumerate(self.rows["Split"].values.tolist()) if x == split] for split in self.rows["Split"].unique()}
        for split in self.splits.keys():
            print(f"[{split}]: {len(self.splits[split])}")
        label_dist = self.rows["Label"].value_counts().sort_index()
        print(f"[dataset] discrete label distribution:\n")
        print(label_dist)
        print()
        print(f"[dataset] required modality: {modal}")
        print(f"[dataset] dataset from {excel_file}, number of cases={len(self.rows)}")

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows["Status"] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df["Event"], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows["Event"].max() + eps
        q_bins[0] = rows["Event"].min() - eps
        disc_labels, q_bins = pd.cut(rows["Event"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), "Label", disc_labels)
        return rows

    def read_WSI(self, wsi):
        if not isinstance(wsi, str):
            # convert nan to empty string
            wsi = ""
        if "WSI" in self.modal and wsi != "":
            wsi = wsi.split(";")
            wsi = [torch.load(os.path.join(self.data_root_wsi, x)) for x in wsi]
            wsi = torch.cat(wsi, dim=0).type(torch.float32)
            return wsi, True
        else:
            wsi = torch.zeros(1, 1).type(torch.float32)
            return wsi, False
        
    def read_Omics(self, omics):
        if not isinstance(omics, str):
            # convert nan to empty string
            omics = ""
        if "Omics" in self.modal and omics != "":
            df = pd.read_csv(os.path.join(self.data_root_omics, omics))
            omics = []
            for signature in self.signatures:
                omic = torch.from_numpy(np.array(df[df["Gene"].isin(signature)]["Value"].values.tolist())).type(torch.float32)
                if omic.size(0) == 0:
                    continue
                omics.append(omic)
            omics = tuple(omics)
            return omics, True
        else:
            omics = [torch.zeros(1).type(torch.float32)]
            return omics, False
    

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI, Omics = case[:6]
        Label = self.rows.iloc[index]["Label"]
        Censorship = 1 if int(Status) == 0 else 0
        WSI, WSI_Flag = self.read_WSI(WSI)
        Omics, Omics_Flag = self.read_Omics(Omics)
        
        return (Study, ID, WSI, WSI_Flag, Omics, Omics_Flag, Event, Censorship, Label)
        
    def __len__(self):
        return len(self.rows)
