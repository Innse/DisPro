#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network.py
@Time    :   2024/01/19 14:55:35
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch
import torch.nn as nn

from .omics_utils import tokenize, \
    load_pretrained_bert, load_pretrained_tokenizer, lock_params
import torch.nn.functional as F
import numpy as np

from .omics_utils import Transformer as gene_encoder

class PromptLearner(nn.Module):
    def __init__(self, tokenizer, token_embedding, ctx_dim, prompt_config, dtype=None):
        super().__init__()

        classnames = prompt_config.classnames
        n_cls = len(classnames)
        n_cls_surv = n_cls // 2
        max_len_ctx = prompt_config.max_len_ctx
        ctx_init = prompt_config.ctx_init
        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            # 设置prefix
            prompt_prefix = ctx_init # str

            # use given words to initialize context vectors
            n_ctx_init = len(ctx_init.split(" "))
            # 补全到max_len_ctx
            words_init = ctx_init.split(" ")
            words_init = words_init + ["X"] * (max_len_ctx - n_ctx_init)
            words_init = " ".join(words_init)
            prompt_init_ids, _ = tokenize(tokenizer, [ctx_init], )
            print("ctx_prompt_init:", words_init)
            print("ctx_prompt_ids_init:", prompt_init_ids)
            prompt_init_ids = torch.tensor(prompt_init_ids)
            with torch.no_grad():
                embedding_init = token_embedding(prompt_init_ids).type(dtype)   # [batch_size, max_len, d_model]
            print("embedding_init: ", embedding_init.shape) # [1, 512, 768]
            ctx_vectors = embedding_init[0, 1: 1 + max_len_ctx, :]
            # repeat context vectors for each surviving class -> shape: (n_cls_surv, max_len_ctx, ctx_dim)
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(n_cls_surv, 1, 1)

        else:
            prompt_prefix = " ".join(["X"] * 64)
            # random initialization
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls_surv, max_len_ctx, ctx_dim, dtype=dtype)
            
            nn.init.normal_(ctx_vectors, std=0.02)
            

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {max_len_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        token_names_ids, token_names_attn = tokenize(tokenizer, classnames, prompt_config.max_len_ctx)
        name_lens = [sum(name) for name in token_names_attn] # len: max_length in tokenize
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts_ids, tokenized_prompts_attn = tokenize(tokenizer, prompts, prompt_config.max_len_ctx)
        tokenized_prompts_ids = torch.tensor(tokenized_prompts_ids)
        tokenized_prompts_attn = torch.tensor(tokenized_prompts_attn)
        
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts_ids).type(dtype)

        self.token_prefix_uncensored = embedding[:4, :1, :].cuda()  # [SOS]
        self.token_prefix_censored = embedding[4:, :1, :].cuda()  # [SOS]
        self.token_suffix_uncensored = embedding[:4, 1:, :].cuda()  # (tokens of described classname), [CLS], [EOS]
        self.token_suffix_censored = embedding[4:, 1:, :].cuda()  # (tokens of described classname), [CLS], [EOS]

        
        self.n_cls = n_cls
        self.max_len_ctx = max_len_ctx
        ctx_attn = torch.ones(tokenized_prompts_attn.shape[:-1]+ (max_len_ctx,))
        self.tokenized_prompts_attn = torch.concat((ctx_attn, tokenized_prompts_attn), dim=-1)
        self.name_lens = name_lens
        self.class_token_position = prompt_config.class_token_position

        self.prompt_config = prompt_config

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.ctx.data)

    def forward(self, c):
        ctx = self.ctx

        
        if self.class_token_position == "end":
            if c == 0:
                prefix = self.token_prefix_uncensored
                suffix = self.token_suffix_uncensored
                
            else:
                prefix = self.token_prefix_censored
                suffix = self.token_suffix_censored
                
            prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.max_len_ctx // 2
            prompts = []
            for i in range(self.n_cls_surv):
                if c == 0:
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                else:
                    name_len = self.name_lens[i + 4]
                    prefix_i = prefix[i + 4: i + 5, :, :]
                    class_i = suffix[i + 4: i + 5, :name_len, :]
                    suffix_i = suffix[i + 4: i + 5, name_len:, :]
                
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls_surv):
                if c == 0:
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                else:
                    name_len = self.name_lens[i + 4]
                    prefix_i = prefix[i + 4: i + 5, :, :]
                    class_i = suffix[i + 4: i + 5, :name_len, :]
                    suffix_i = suffix[i + 4: i + 5, name_len:, :]
                
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class ClassStrEncoder(nn.Module):
    def __init__(self, clsStr_encoder, out_embed_dim=512):
        super().__init__()
        self.clsStr_encoder = clsStr_encoder
        self.transformer = clsStr_encoder.encoder
        self.text_projection = nn.Parameter(torch.empty(self.clsStr_encoder.width, out_embed_dim))
        self.dtype = clsStr_encoder.dtype
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.text_projection.data)

    def forward(self, prompts, tokenized_prompts_attn):
        extended_attention_mask = self.clsStr_encoder.get_extended_attention_mask(tokenized_prompts_attn, tokenized_prompts_attn.shape)
        extended_attention_mask = extended_attention_mask.to(prompts.device)

        head_mask = self.clsStr_encoder.get_head_mask(None, self.clsStr_encoder.config.num_hidden_layers)
        
        
        x = self.transformer(prompts, attention_mask=extended_attention_mask,
                             head_mask=head_mask,
                             output_attentions= self.clsStr_encoder.config.output_attentions,
                             output_hidden_states=True,
                             return_dict=self.clsStr_encoder.config.use_return_dict)['hidden_states'][-1]
        x = x[:, 0, :]

        x =  x @ self.text_projection

        return x



class CustomCLIP(nn.Module):
    def __init__(self, modal_enc_name, tokenizer, clsStr_encoder, prompt_config, gene_config):
        super().__init__()
        self.dtype = clsStr_encoder.dtype
        self.pooler = gene_config.pooler

        # define prompt
        self.prompt_learner = PromptLearner(ctx_dim=clsStr_encoder.encoder.layer[-1].output.LayerNorm.weight.shape[0],
                                            tokenizer=tokenizer, 
                                            token_embedding=clsStr_encoder.embeddings,
                                            dtype=self.dtype, prompt_config=prompt_config)
        self.tokenized_prompts_attn = self.prompt_learner.tokenized_prompts_attn

        # define encoder for each tower
        if modal_enc_name == 'PathTransMean':
            model_dict = {"omic_sizes": gene_config.omics_size, "dropout": 0.25, "num_classes": gene_config.num_classes, "pooler": "mean"}
            self.modality_encoder = gene_encoder(**model_dict)
        self.clsStr_encoder = ClassStrEncoder(clsStr_encoder, out_embed_dim=gene_config.out_embed_dim)
        
        self.dropout = nn.Dropout(gene_config.drop_rate)
        self.projection = nn.Linear(gene_config.in_embed_dim, gene_config.out_embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, x, c):
        # combine embeddings of clsStr
        prompts = self.prompt_learner(c)
        if c == 0:
            tokenized_prompts_attn = self.tokenized_prompts_attn[:4]
        else:
            tokenized_prompts_attn = self.tokenized_prompts_attn[4:]
        
        
        # get clsStr features
        clsStr_features = self.clsStr_encoder(prompts, tokenized_prompts_attn)
        clsStr_features = F.normalize(clsStr_features, dim=-1)

        # get modal features

        x = self.modality_encoder(**x)

        if self.pooler == "mean":
            x = torch.mean(x, dim=1)
        elif self.pooler == "first":
            x = x[:, 0]
        elif self.pooler == "last":
            x = x[:, -1]
        elif self.pooler == "cls":
            x = x[:, 0]

        x = self.dropout(x)
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        
        # get logit mtx
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * x @ clsStr_features.t()
        

        return logits


class CoOp(nn.Module):
    def __init__(self, clsStrEnc_name, modal_enc_name, prompt_config, gene_config) -> None:
        super().__init__()
        self.prompt_config = prompt_config
        self.tokenizer = load_pretrained_tokenizer(clsStrEnc_name)
        self.clsStr_encoder = load_pretrained_bert(clsStrEnc_name)
        self.clsStr_encoder = lock_params(self.clsStr_encoder)

        self.model = CustomCLIP(tokenizer=self.tokenizer, 
                                clsStr_encoder=self.clsStr_encoder,
                                prompt_config=prompt_config, 
                                modal_enc_name=modal_enc_name,
                                gene_config=gene_config)
    
    def forward(self, x, c):
        return self.model(x, c)
