#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_utils.py
@Time    :   2024/01/19 14:58:20
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
import torch

def load_pretrained_tokenizer(model_name):
    if 'gpt' in str.lower(model_name):
        from transformers import BioGptTokenizer
        tokenizer = BioGptTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=False)
    
    if not 'cls_token' in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            
    if not 'sep_token' in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        
    return tokenizer

def load_pretrained_bert(model_name = 'emilyalsentzer/Bio_ClinicalBERT', tokenizer=None):
    assert model_name in ['microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 
                   'emilyalsentzer/Bio_ClinicalBERT','dmis-lab/biobert-base-cased-v1.2',
                   'yikuan8/Clinical-Longformer']
    config = AutoConfig.from_pretrained(model_name)

    if model_name == 'yikuan8/Clinical-Longformer':
        model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                config = config).longformer
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                    config = config).bert
        model.width = model.embeddings.word_embeddings.embedding_dim
        model.vocab_size = model.embeddings.word_embeddings.num_embeddings
        model.autoregressive = False
        model.pool = 'cls'
    return model
    
def load_pretrained_gpt(model_name = 'microsoft/biogpt', tokenizer=None):
    assert model_name in ['microsoft/biogpt']

    if model_name == 'microsoft/biogpt':
        from transformers import BioGptForCausalLM
        model = BioGptForCausalLM.from_pretrained(model_name).biogpt
        embed_tokens = model.embed_tokens
        if tokenizer:
            print('len of tokenizer: ', len(tokenizer))
            embed_tokens = model.resize_token_embeddings(len(tokenizer))

        encoder = model.layers
        embeddings = [embed_tokens, model.embed_positions]
    else:
        raise NotImplementedError
    return model, encoder, embeddings

def lock_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
def tokenize(tokenizer, texts, max_length=512):
    tokens = tokenizer.batch_encode_plus(texts,
                                         max_length=max_length,
                                         # Add '[CLS]' and '[SEP]'
                                         add_special_tokens=True,
                                         return_token_type_ids=False,
                                         truncation=True,
                                         padding='max_length',
                                         return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(nn.Linear(dim1, dim2), nn.ELU(), nn.AlphaDropout(p=dropout, inplace=False))


class Transformer(nn.Module):
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        dropout=0.25,
        num_classes=4,
        pooler="mean",
    ):
        super(Transformer, self).__init__()
        
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.pooler = pooler
        assert pooler in ["mean", "first", "last", "cls"]
        if pooler == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
            # nn.init.normal_(self.cls_token, std=1e-6)
            nn.init.xavier_normal_(self.cls_token.data)
        # omic embedding for each pathway
        self.init_per_path_model(omic_sizes)

        # transformer
        trans_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=512, dropout=self.dropout, activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)

        # classification layer
        self.classifier = nn.Linear(768, num_classes)

        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

        # init per-pathway models
        for sig_network in self.sig_networks:
            for layer in sig_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def init_per_path_model(self, omic_sizes):
        hidden = [768, 768]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def forward(self, **kwargs):
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, self.num_pathways + 1)]
        # ---> get pathway embeddings
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()).squeeze() for idx, sig_feat in enumerate(x_omic)]  ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  ### omic embeddings are stacked

        if self.pooler == "cls":
            cls_token = self.cls_token.expand(1, -1, -1).cuda()
            h_omic_bag = torch.cat((cls_token, h_omic_bag), dim=1)
        # ---> transformer
        h_omic_bag = self.transformer(h_omic_bag)

        return h_omic_bag