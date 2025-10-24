#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network.py
@Time    :   2024/09/25 21:22:57
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import random
from urllib.request import ProxyBasicAuthHandler
import torch
import torch.nn.functional as F
import torch.nn as nn
from .wsi_utils import topj_pooling, avg_pooling
from .model_utils import Attn_Net_Gated, MultiheadAttention
from .utils import load_pretrained_model, load_uni_models
import numpy as np

class Transformer(nn.Module):
    def __init__(self, unis_config, omic_sizes, encoder, num_classes, max_length=512, n_WSI=1024, dim_token=128, fine_tune=False) -> None:
        super(Transformer, self).__init__()
        assert encoder in ["PubMedBERT", "ClinicalBERT", "BioBERT", "LongFormer"], "encoder must be one of ['PubMedBERT', 'ClinicalBERT', 'BioBERT', 'LongFormer']"
        checkpoints = {
            "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
            "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
            "LongFormer": "yikuan8/Clinical-Longformer",
        }
        self.unis_config = unis_config
        self.max_length = max_length
        self.dim_token = dim_token
        print('[Max Length]: ', self.max_length)
        print('[Dim_token]: ', self.dim_token)
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_token))
        if len(omic_sizes) < int(self.max_length // 2):
            self.len_omic = len(omic_sizes)
        else:
            self.len_omic = int(self.max_length // 2)
        self.len_path = self.max_length - self.len_omic - 1 # 1 for cls token

        print('[Len Path]: ', self.len_path)
        print('[Len Omic]: ', self.len_omic)
        
        self.holder_path = nn.Parameter(torch.randn(1, self.len_path, dim_token))
        self.holder_omic = nn.Parameter(torch.randn(1, self.len_omic, dim_token))


        self.uni_model_wsi, self.uni_model_omics = load_uni_models(unis_config["path_model_wsi"], 
                                                         unis_config["path_model_omics"], 
                                                         omic_sizes, num_classes, 
                                                         unis_config["prompt_config"], 
                                                         unis_config["wsi_config"], 
                                                         unis_config["gene_config"])
        
        self.uni_model_wsi = self.lock_params(self.uni_model_wsi)
        self.uni_model_omics = self.lock_params(self.uni_model_omics)

        self.setup_unidistill_wsi()
        self.setup_unidistill_omics()


        self.wsi_adapter = self.unlock_params(self.uni_model_wsi.model.modality_encoder)
        self.omics_adapter = self.unlock_params(self.uni_model_omics.model.modality_encoder)

        
        self.wsi_attn_head = Attn_Net_Gated(L = 768, D = 768, n_classes = 1, dropout = True)
        self.omics_attn_head = Attn_Net_Gated(L = 768, D = 768, n_classes = 1, dropout = True)
        

        # transformer layer
        self.Transformer = load_pretrained_model(checkpoints[encoder]).encoder
        
        
        # linear head
        self.linear_head = nn.Sequential(
                                nn.Linear(self.dim_token, 512),
                                nn.ReLU())
        # classification layer
        self.classifier = nn.Linear(512, self.num_classes)
        self.init_weights()
        if not fine_tune:
            self.Transformer = self.lock_params(self.Transformer)
    
    def lock_params(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def unlock_params(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.cls_token.data)
        torch.nn.init.xavier_uniform_(self.holder_path.data)
        torch.nn.init.xavier_uniform_(self.holder_omic.data)
        torch.nn.init.xavier_uniform_(self.linear_head[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.linear_head[0].bias)
        torch.nn.init.zeros_(self.classifier.bias)
    
    
    
    def trunk_tokens(self, inputs, max_length, method="random"):
        '''
        inputs: tensor, shape (bs, seq_len, dim)
        '''
        len_token = inputs.shape[1]
        if len_token > max_length:
            if method == "random":
                idxes = random.sample(range(len_token), max_length)
                inputs = inputs[:, idxes, :]
            elif method == "head":
                inputs = inputs[:, : max_length, :]
            elif method == "tail":
                inputs = inputs[:, -max_length:, :]
            else:
                raise NotImplementedError("trunk method not implemented.")
                
        return inputs
    
    def setup_unidistill_wsi(self):
        self.wsi_prompt_learner = self.uni_model_wsi.model.prompt_learner
        self.wsi_tokenized_prompts_attn = self.uni_model_wsi.model.tokenized_prompts_attn
        self.wsi_clsStr_encoder = self.uni_model_wsi.model.clsStr_encoder
        self.wsi_logit_scale = self.uni_model_wsi.model.logit_scale
        
        

    def setup_unidistill_omics(self):
        self.omics_prompt_learner = self.uni_model_omics.model.prompt_learner
        self.omics_tokenized_prompts_attn = self.uni_model_omics.model.tokenized_prompts_attn
        self.omics_clsStr_encoder = self.uni_model_omics.model.clsStr_encoder
        self.omics_logit_scale = self.uni_model_omics.model.logit_scale

    def distill_wsi(self, tokens_path, data_censor):
        prompts_wsi = self.wsi_prompt_learner(data_censor)
        
        if data_censor == 0:
            tokenized_prompts_attn_wsi = self.wsi_tokenized_prompts_attn[:4]
        else:
            tokenized_prompts_attn_wsi = self.wsi_tokenized_prompts_attn[4:]
        clsStr_features_wsi = self.wsi_clsStr_encoder(prompts_wsi, tokenized_prompts_attn_wsi)
        clsStr_features_wsi = F.normalize(clsStr_features_wsi, dim=-1)

        
        features_path = F.normalize(tokens_path, dim=-1)

        logit_scale_wsi = self.wsi_logit_scale.exp()
        logits_wsi = (logit_scale_wsi * features_path @ clsStr_features_wsi.t()).squeeze()


        if self.unis_config["wsi_config"].pool_method == 'topj':
            logit_wsi = topj_pooling(logits_wsi, self.unis_config["wsi_config"].topj) #, use_batch=self.args.use_batch)
        elif self.unis_config["wsi_config"].pool_method == 'avg':
            logit_wsi = avg_pooling(logits_wsi)
        else:
            raise NotImplementedError("WSI Pooling method not implemented.")
        
        return logit_wsi, logits_wsi

    def distill_omics(self, tokens_omic, data_censor):
        prompts_omic = self.omics_prompt_learner(data_censor)
        
        if data_censor == 0:
            tokenized_prompts_attn_omic = self.omics_tokenized_prompts_attn[:4]
        else:
            tokenized_prompts_attn_omic = self.omics_tokenized_prompts_attn[4:]
        clsStr_features_omic = self.omics_clsStr_encoder(prompts_omic, tokenized_prompts_attn_omic)
        clsStr_features_omic = F.normalize(clsStr_features_omic, dim=-1)

        x_omic = F.normalize(tokens_omic, dim=-1)

        logit_scale_omic = self.omics_logit_scale.exp()
        logits_omics = (logit_scale_omic * x_omic @ clsStr_features_omic.t()).squeeze()

        if self.unis_config["gene_config"].pool_method == 'topj':
            logit_omics = topj_pooling(logits_omics, self.unis_config["gene_config"].topj)
        elif self.unis_config["gene_config"].pool_method == 'avg':
            logit_omics = avg_pooling(logits_omics)
        else:
            raise NotImplementedError("Omics Pooling method not implemented.")
        return logit_omics, logits_omics
    
    def token_scoring(self, tokens, modal, data_censor, label=None):
        if modal == 'WSI':
            modal_attn_head = self.wsi_attn_head
            len_token = self.len_path
        elif modal == 'Omics':
            modal_attn_head = self.omics_attn_head
            len_token = self.len_omic
        else:
            raise NotImplementedError("Modal not implemented.")
        
        if tokens.shape[1] <= len_token:
            return tokens, torch.ones(tokens.shape[1])
        attn, _ = modal_attn_head(tokens.squeeze()) # (seq_len, 1)
        attn = torch.transpose(attn, 0, 1) # (1, seq_len)
        self_scores = torch.sigmoid(attn).squeeze()

        
        if label is None:
            # utilize unimodel to score tokens
            _, logits_wsi_c = self.distill_wsi(tokens, 0)
            _, logits_wsi_uc = self.distill_wsi(tokens, 1)
            _, logits_omics_c = self.distill_omics(tokens, 0)
            _, logits_omics_uc = self.distill_omics(tokens, 1)
            logits_wsi = torch.concat((logits_wsi_c, logits_wsi_uc), dim=1)
            logits_omics = torch.concat((logits_omics_c, logits_omics_uc), dim=1)

            # predict label
            sim_wsi = torch.sigmoid(logits_wsi)
            sim_omics = torch.sigmoid(logits_omics)
            sim_mtx = sim_wsi + sim_omics
            scores_topj = topj_pooling(sim_mtx, 256)
            label = torch.argmax(scores_topj, dim=1)
        else:
            _, logits_wsi = self.distill_wsi(tokens, data_censor)
            _, logits_omics = self.distill_omics(tokens, data_censor)
            
        
        logits_wsi = logits_wsi[:, label]
        logits_omics = logits_omics[:, label]
        wsi_scores = torch.sigmoid(logits_wsi).squeeze()
        omics_scores = torch.sigmoid(logits_omics).squeeze()
        
        
        # linearly combine scores
        scores = self_scores + wsi_scores + omics_scores
        # get index of topj tokens
        indices = torch.topk(scores, len_token, dim=0)[1]
        tokens_selected = torch.index_select(tokens, 1, indices)
        
        return tokens_selected, scores

    def forward(self, **kwargs):
        bs = 1
        device = self.cls_token.device
        data_censor = kwargs["censor"]
        label = kwargs["label"]

        cls_token = self.cls_token.repeat(bs, 1, 1).to(device)
        tokens = cls_token
        tokens_mask = torch.ones(bs, 1).to(device)

        # construct tokens for each modality
        #
        data_WSI = kwargs["x_WSI"]
        scores_path = None
        if data_WSI is not None:
            tokens_path  = self.wsi_adapter(data_WSI)
            tokens_path, scores_path = self.token_scoring(tokens_path, 'WSI', data_censor, label)
            tokens_mask = torch.cat((tokens_mask, torch.ones(bs, tokens_path.shape[1]).to(device)), dim=1)
            len_real_path = tokens_path.shape[1]
            if len_real_path < self.len_path:
                wsi_placeholder = torch.zeros(bs, self.len_path - tokens_path.shape[1], self.dim_token).to(device)
                tokens_path = torch.cat((tokens_path, wsi_placeholder), dim=1)
                tokens_mask = torch.cat((tokens_mask, torch.zeros(bs, self.len_path - len_real_path).to(device)), dim=1)
            tokens_path = (tokens_path + self.holder_path.repeat(bs, 1, 1)) / 2.0
            tokens = torch.cat((tokens, tokens_path), dim=1)
        # when wsi is missing
        else:
            wsi_placeholder = self.holder_path.repeat(bs, 1, 1)
            tokens = torch.cat((tokens, wsi_placeholder), dim=1)
            tokens_mask = torch.cat((tokens_mask, torch.ones(bs, self.len_path).to(device)), dim=1) # zero -> one
        
        #
        data_Omics = kwargs["x_Omics"]
        scores_omic = None
        if data_Omics is not None:
            tokens_omic = self.omics_adapter(**data_Omics)
            tokens_omic, scores_omic = self.token_scoring(tokens_omic, 'Omics', data_censor, label)
            tokens_mask = torch.cat((tokens_mask, torch.ones(bs, tokens_omic.shape[1]).to(device)), dim=1)
            len_real_omic = tokens_omic.shape[1]
            if len_real_omic < self.len_omic:
                omic_placeholder = torch.zeros(bs, self.len_omic - len_real_omic, self.dim_token).to(device)
                tokens_omic = torch.cat((tokens_omic, omic_placeholder), dim=1)
                tokens_mask = torch.cat((tokens_mask, torch.zeros(bs, self.len_omic - len_real_omic).to(device)), dim=1)
            tokens_omic = (tokens_omic + self.holder_omic.repeat(bs, 1, 1)) / 2.0
            tokens = torch.cat((tokens, tokens_omic), dim=1)
        # when omics is missing
        else:
            omic_placeholder = self.holder_omic.repeat(bs, 1, 1)
            tokens = torch.cat((tokens, omic_placeholder), dim=1)
            tokens_mask = torch.cat((tokens_mask, torch.ones(bs, self.len_omic).to(device)), dim=1)
        
        
        outs_llm = self.Transformer(tokens, attention_mask=tokens_mask, output_attentions=True)
        tokens = outs_llm['last_hidden_state']
        attns = outs_llm['attentions']
        
        
        # ---> distill unimodal knowledge for missing modality
        if data_WSI is None:
            tokens_path = tokens[:, 1: self.len_path + 1, :]
            logits_wsi, _ = self.distill_wsi(tokens_path, data_censor)
            hazards_wsi = torch.sigmoid(logits_wsi)
            S_wsi = torch.cumprod(1 - hazards_wsi, dim=1)

        if data_Omics is None:
            tokens_omic = tokens[:, self.len_path + 1:, :]
            logits_omics, _ = self.distill_omics(tokens_omic, data_censor)
            hazards_omics = torch.sigmoid(logits_omics)
            S_omics = torch.cumprod(1 - hazards_omics, dim=1)

        # fetch cls token for each modality
        feature = tokens[:, 0, :]
        feature = self.linear_head(feature)
        # prediction
        logits = self.classifier(feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return_values = {"hazards": hazards, "S": S, 
                         "attns": attns, 
                         "scores_path": scores_path, "scores_omics": scores_omic}
        if data_WSI is None:
            return_values["hazards_wsi"] = hazards_wsi
            return_values["S_wsi"] = S_wsi
        
        if data_Omics is None:
            return_values["hazards_omics"] = hazards_omics
            return_values["S_omics"] = S_omics

        return return_values
