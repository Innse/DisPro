#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_utils.py
@Time    :   2024/09/07 16:23:55
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch.nn as nn
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

class WSI_head(nn.Module):
    def __init__(self, modal_config):
        super().__init__()
        head_layers = OrderedDict()
        if modal_config.modal_enc_name == 'fc':
            head_layers['proj'] = nn.Linear(modal_config.in_embed_dim, modal_config.out_embed_dim)
            head_layers['relu'] = nn.ReLU()

        self.head = nn.Sequential(head_layers)

    def forward(self, img_feature):
        return self.head(img_feature)


def load_pretrained_tokenizer(model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast=False)
    
    if not 'cls_token' in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            
    if not 'sep_token' in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        
    return tokenizer

def load_pretrained_bert(model_name = 'dmis-lab/biobert-base-cased-v1.2'):
    config = AutoConfig.from_pretrained(model_name)

    model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                    config = config).bert
    model.width = model.embeddings.word_embeddings.embedding_dim
    model.vocab_size = model.embeddings.word_embeddings.num_embeddings
    model.autoregressive = False
    model.pool = 'cls'
    return model


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

def topj_pooling(logits, topj): #, use_batch=False):
    """
    logits: N x 1 logit for each patch
    coords: N x 2 coordinates for each patch
    topj: tuple of the top number of patches to use for pooling
    ss: spatial smoothing by k-nn
    ss_k: k in k-nn for spatial smoothing
    """
    
    maxj = min(topj, logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True)
    
    logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
    return logits

def avg_pooling(logits):
    # return {-1: logits.mean(dim=0, keepdim=True).argmax(dim=1)} # preds
    return logits.mean(dim=0, keepdim=True) # logits