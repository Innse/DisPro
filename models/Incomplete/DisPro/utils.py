#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/01/19 14:58:20
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
"""

# Here put the import lib
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from .CoOp_WSI import CoOp as CoOp_WSI
from .CoOp_Omics import CoOp as CoOp_Omics

def load_pretrained_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    if not "cls_token" in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    if not "sep_token" in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})

    return tokenizer


def load_pretrained_model(model_name="emilyalsentzer/Bio_ClinicalBERT", tokenizer=None):
    assert model_name in ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-base-cased-v1.2", "yikuan8/Clinical-Longformer"]
    config = AutoConfig.from_pretrained(model_name)

    if model_name == "yikuan8/Clinical-Longformer":
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config).longformer
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config).bert
        model.width = model.embeddings.word_embeddings.embedding_dim
        model.vocab_size = model.embeddings.word_embeddings.num_embeddings
        model.autoregressive = False
        model.pool = "cls"
    return model

def lock_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_uni_models(path_model_wsi, path_model_omics, omics_size, num_classes, prompt_configs, wsi_config, gene_config):
    ckpt_wsi = torch.load(path_model_wsi)
    ckpt_omics = torch.load(path_model_omics)

    prompt_config_wsi = prompt_configs['WSI']
    prompt_config_omics = prompt_configs['Omics']
    
    wsi_model_dict = {"clsStrEnc_name": "dmis-lab/biobert-base-cased-v1.2",
                          "prompt_config": prompt_config_wsi, "modal_config": wsi_config}
    model_wsi = CoOp_WSI(**wsi_model_dict)
    model_wsi.load_state_dict(ckpt_wsi['state_dict'])
    

    gene_config.omics_size = omics_size
    gene_config.num_classes = num_classes
    omics_model_dict = {"clsStrEnc_name": "dmis-lab/biobert-base-cased-v1.2",
                          "modal_enc_name": "PathTransMean", "prompt_config": prompt_config_omics, "gene_config": gene_config}
    model_omics = CoOp_Omics(**omics_model_dict)
    model_omics.load_state_dict(ckpt_omics['state_dict'])
    

    return model_wsi, model_omics