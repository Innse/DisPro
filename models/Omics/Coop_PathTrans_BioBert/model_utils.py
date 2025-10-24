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

def topj_pooling(logits, topj): #, use_batch=False):
    """
    logits: N x 1 logit for each patch
    coords: N x 2 coordinates for each patch
    topj: tuple of the top number of patches to use for pooling
    ss: spatial smoothing by k-nn
    ss_k: k in k-nn for spatial smoothing
    """
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True)
    
    logits = {j : values[:min(j, maxj)].sum(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    logits = {key: val for key,val in logits.items()} # dict of predicted class logits

    return logits

def avg_pooling(logits):
    return {-1: logits.mean(dim=0, keepdim=True)} # logits