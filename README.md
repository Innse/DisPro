<div align="center">

<h1><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Distilled_Prompt_Learning_for_Incomplete_Multimodal_Survival_Prediction_CVPR_2025_paper.html">Distilled Prompt Learning for Incomplete Multimodal Survival Prediction</a></h1>

**Yingxue Xu, Fengtao Zhou, Chenyu Zhao, Yihui Wang, Can Yang, Hao Chen** 

![](https://komarev.com/ghpvc/?username=InnseDisPro&label=visitors)
![GitHub stars](https://badgen.net/github/stars/Innse/DisPro)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2503.01653-b31b1b.svg)](https://arxiv.org/abs/2503.01653)

</div>

### Share us a :star: if this repo does help

This repository is the official implementation of *DisPro*. If you encounter any question, please feel free to contact us. 
You can create an issue or just send email to me (yxueb@connect.ust.hk). Also welcome for any idea exchange and discussion.


## Table of Contents
- [Introduction](#introduction)
- [Data preparation](#data-preparation)
- [Requirements](#requirements)
- [Run](#run)
- [License & Citation](#license--citation)

## Introduction

![DisPro](./imgs/figure1_dispro.png)

The integration of multimodal data including pathology images and gene profiles is widely applied in precise survival prediction. Despite recent advances in multimodal survival models, collecting complete modalities for multimodal fusion still poses a significant challenge, hindering their application in clinical settings. Current approaches tackling incomplete modalities often fall short, as they typically compensate for only a limited part of the knowledge of missing modalities. To address this issue, we propose a Distilled Prompt Learning framework (DisPro) to utilize the strong robustness of Large Language Models (LLMs) to missing modalities, which employs two-stage prompting for compensation of comprehensive information for missing modalities. In the first stage, Unimodal Prompting (UniPro) distills the knowledge distribution of each modality, preparing for supplementing modality-specific knowledge of the missing modality in the subsequent stage. In the second stage, Multimodal Prompting (MultiPro) leverages available modalities as prompts for LLMs to infer the missing modality, which provides modality-common information. Simultaneously, the unimodal knowledge acquired in the first stage is injected into multimodal inference to compensate for the modality-specific knowledge of the missing modality. Extensive experiments covering various missing scenarios demonstrated the superiority of the proposed method.


## Data preparation
### WSIs
1. Preprocessing WSI data by [PrePATH](https://github.com/birkhoffkiki/PrePATH) and extract `uni` features (or other foundation features you want) for each slide. PrePATH provides an easy-to-use tool for WSI preprocessing.
2. Set up the dir of feature as `data_root_wsi`.

### Gene
1. The pathways signature is provided in the `datasets` folder.
2. The RNA-Seq expression data are provided on [GoogleDrive](https://drive.google.com/drive/folders/18cxpThdOMgX_BWvsn5i-DTYdTkECcHsx?usp=sharing). You can unzip them and put it somewhere you like, which should be set as `data_root_omics`.

### CSV
- The data file for complete modaltiy is provided in `splits/[STUDY]_Splits.csv`.

- The data file for various missing scenarios are provided in `splits/csv_missing_cleaned` folder, where the number following 'W' or 'O' represents the missing rate for WSI or Omics, respectively. (Or you can simulate the missing scenarios by yourself.)


## Requirements

1. Create a new conda environmenty.
```
conda create -n dispro python=3.10
conda activate dispro
```
2. Install the required packages.
```
torch == 2.3.0+cu121
timm == 0.9.8
torchvision == 0.18.0
numpy == 1.24.3
```
or directly install environment by `yaml` file.
```
conda create -n dispro -f dispro.yaml
```
 


## Run
There are two stages in DisPro.
### Stage 1 - UniPro
You need to train a UniPro for each modality. 
#### WSI
To train the UniPro for pathology, you can specify the arguments in the `run_unipro_wsi.sh` script stored in [scripts](./scripts/) and run it.
```bash
bash scripts/run_unipro_wsi.sh
```
#### Omics
To train the UniPro for pathology, you can specify the arguments in the `run_unipro_wsi.sh` script stored in [scripts](./scripts/) and run it.
```bash
bash scripts/run_unipro_omics.sh
```

### Stage 2 - MultiPro
1. You need to specify the save path `result_root` to checkpoints in `utils/get_path_ckpt_dict.py`.

2. You need to get the json file storing paths of ckpt of UniPro for every modality by run:
```
python utils/get_path_ckpt_dict.py
```


3. To train the MultiPro for missing modality, you can specify the arguments in the `run_multipro.sh` script stored in [scripts](./scripts/) and run it.
```bash
bash scripts/run_multipro.sh
```



## License & Citation
This project is licensed under the Apache-2.0 License.

If you find this work useful, please cite our paper:
```
@InProceedings{Xu_2025_CVPR,
    author    = {Xu, Yingxue and Zhou, Fengtao and Zhao, Chenyu and Wang, Yihui and Yang, Can and Chen, Hao},
    title     = {Distilled Prompt Learning for Incomplete Multimodal Survival Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5102-5111}
}
```
