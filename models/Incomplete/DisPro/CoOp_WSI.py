#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network.py
@Time    :   2024/01/24 16:23:08
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch
import torch.nn as nn
from .wsi_utils import load_pretrained_bert, tokenize, \
                load_pretrained_tokenizer, lock_params, \
                topj_pooling, avg_pooling
import numpy as np
import torch.nn.functional as F


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

        #
        # max_len for prefix+cls_name
        max_len_cls = 512 - prompt_config.max_len_ctx
        classnames = [name.replace("_", " ") for name in classnames]
        token_names_ids, token_names_attn = tokenize(tokenizer, classnames, max_len_cls)
        name_lens = [sum(name) for name in token_names_attn] # len: max_length in tokenize
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # ids for classname description
        tokenized_prompts_ids, tokenized_prompts_attn = tokenize(tokenizer, prompts, max_len_cls)
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
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.clsStr_encoder.width ** -0.5)

    def forward(self, prompts, tokenized_prompts_attn):
        extended_attention_mask = self.clsStr_encoder.get_extended_attention_mask(tokenized_prompts_attn, tokenized_prompts_attn.shape)
        head_mask = self.clsStr_encoder.get_head_mask(None, self.clsStr_encoder.config.num_hidden_layers)
        
        x = self.transformer(prompts, attention_mask=extended_attention_mask.cuda(),
                             head_mask=head_mask,
                             output_attentions= self.clsStr_encoder.config.output_attentions,
                             output_hidden_states=True,
                             return_dict=self.clsStr_encoder.config.use_return_dict)['hidden_states'][-1]
        x = x[:, 0, :]
        
        x =  x @ self.text_projection

        return x



class CustomCLIP(nn.Module):
    def __init__(self, tokenizer, clsStr_encoder, prompt_config, modal_config):
        super().__init__()
        self.dtype = clsStr_encoder.dtype
        self.modal_config = modal_config

        self.prompt_learner = PromptLearner(ctx_dim=clsStr_encoder.encoder.layer[-1].output.LayerNorm.weight.shape[0],
                                            tokenizer=tokenizer, 
                                            token_embedding=clsStr_encoder.embeddings,
                                            dtype=self.dtype, prompt_config=prompt_config)
        self.tokenized_prompts_attn = self.prompt_learner.tokenized_prompts_attn

        # define encoder for each tower
        if modal_config.modal_enc_name == 'fc':
            from .wsi_utils import WSI_head as modal_encoder
            self.modality_encoder = modal_encoder(modal_config)

        
        
        self.clsStr_encoder = ClassStrEncoder(clsStr_encoder, out_embed_dim=modal_config.out_embed_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        

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
        x = x.type(self.dtype)
        x = self.modality_encoder(x)
        x = F.normalize(x, dim=-1)
        
        # get logit mtx
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * x @ clsStr_features.t()).squeeze()
        
        if self.modal_config.pool_method == 'topj':
            logits = topj_pooling(logits, self.modal_config.topj) #, use_batch=self.args.use_batch)
        elif self.modal_config.pool_method == 'avg':
            logits = avg_pooling(logits)
        else:
            raise NotImplementedError
        
        return logits


class CoOp(nn.Module):
    def __init__(self, clsStrEnc_name, prompt_config, modal_config) -> None:
        super().__init__()
        self.prompt_config = prompt_config
        self.tokenizer = load_pretrained_tokenizer(clsStrEnc_name)
        self.clsStr_encoder = load_pretrained_bert(clsStrEnc_name)
        self.clsStr_encoder = lock_params(self.clsStr_encoder)

        self.model = CustomCLIP(tokenizer=self.tokenizer, 
                                clsStr_encoder=self.clsStr_encoder,
                                prompt_config=prompt_config,
                                modal_config=modal_config)
    
    
    def forward(self, x, c):
        return self.model(x, c)
        