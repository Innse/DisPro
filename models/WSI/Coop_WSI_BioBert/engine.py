#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   engine.py
@Time    :   2024/01/19 18:13:30
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import os
import numpy as np

import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        
        # create result directory
        if not os.path.exists(os.path.join(results_dir, "fold_" + str(fold))):
            os.makedirs(os.path.join(results_dir, "fold_" + str(fold)))
        
        self.best_scores = 0
        self.best_epoch = 0
        self.filename_best = None
        self.results = {
            "wsi": {"c-index": 0, "checkpoint": None, "epoch": 0},
        }
    
    def learning(self, model, dataloaders, criterion, optimizer, scheduler):
        
        # cuda
        device_count = torch.cuda.device_count()
        if torch.cuda.is_available() and device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            model = nn.DataParallel(model)
        
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(dataloaders['val'], model, criterion)
            return
        
        
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(dataloaders['train'], model, criterion, optimizer)
            # evaluate on validation set
            scores = self.validate(dataloaders['val'], model, criterion)
            
            is_best = scores > self.best_scores
            if is_best:
                self.best_scores = scores
                self.best_epoch = self.epoch
                self.save_checkpoint({"epoch": epoch, "state_dict": model.state_dict()})
            scheduler.step()
            print(">>>")
            print(">>>")

        self.results["wsi"]["c-index"] = 0.0
        self.results["wsi"]["epoch"] = 0
        return self.results

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        if self.args.tqdm:
            from tqdm import tqdm
            dataloader = tqdm(data_loader, desc="Train Epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------Train Epoch {}-------------------------------".format(self.epoch))

        
        
        for batch_idx, (data_Study, data_ID, data_WSI, WSI_Flag, data_Omics, Omics_Flag, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                c = data_Censorship.type(torch.FloatTensor).cuda()
                Y = data_Label.type(torch.LongTensor).cuda()
            
            
            # prediction
            logits = model(x=data_WSI, c=c)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = criterion(hazards=hazards, S=S, Y=Y, c=c)

            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            
            # backward to update parameters
            loss.backward()
            # print(model.model.prompt_learner.ctx.grad)
            optimizer.step()
            optimizer.zero_grad()
            
            # results
            loss_value = loss.item()
            
            
            if (batch_idx + 1) % 20 == 0:
                out_str = 'Epoch [{}] batch {}, loss: {:.4f}'.format(
                    self.epoch, batch_idx, loss_value)
                print(out_str)
        

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        
        print('Epoch: {}, loss: {:.4f}, train_cindex: {:.4f}'.format(
                self.epoch, loss, c_index))
        
    
    def validate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))

        if self.args.tqdm:
            from tqdm import tqdm
            dataloader = tqdm(data_loader, desc="Test Epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------Test Epoch {}-------------------------------".format(self.epoch))

        
        with torch.no_grad():
            for batch_idx, (data_Study, data_ID, data_WSI, WSI_Flag, data_Omics, Omics_Flag, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
                if torch.cuda.is_available():
                    data_WSI = data_WSI.cuda()
                    c = data_Censorship.type(torch.FloatTensor).cuda()
                    Y = data_Label.type(torch.LongTensor).cuda()
                
                # prediction
                logits = model(x=data_WSI, c=c)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = criterion(hazards=hazards, S=S, Y=Y, c=c)

                
                total_loss += loss.item()

                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = data_Censorship.item()
                all_event_times[batch_idx] = data_Event

                
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        
        
        print('Epoch: {}, loss: {:.4f}, val_cindex: {:.4f}'.format(
                self.epoch, loss, c_index))
        
        return c_index
    
    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, "fold_" + str(self.fold), "model_best_{score:.4f}_{epoch}.pth.tar".format(score=state["best_score"], epoch=state["epoch"]))
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)