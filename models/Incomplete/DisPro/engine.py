import os
import numpy as np
from tqdm import tqdm
import joblib
from sksurv.metrics import concordance_index_censored

import torch.optim

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        
        self.best_scores = 0
        self.best_epoch = 0
        self.filename_bests = {}
        self.results = {
            "wsi": {"c-index": 0, "checkpoint": None, "epoch": 0},
            "omics": {"c-index": 0, "checkpoint": None, "epoch": 0},
            "wsi-omics": {"c-index": 0, "checkpoint": None, "epoch": 0},
        }
        self.eval_set = self.args.eval_settings.split("_")

    def learning(self, model, dataloaders, criterion, optimizer, scheduler):
        if self.args.multi_lr:
            optimizer = torch.optim.Adam([
                {'params': model.holder_path.data, 'lr': 2e-4},
                {'params': model.holder_omic.data, 'lr': 1e-5},
                {'params': model.wsi_adapter.parameters(), 'lr': 2e-4},
                {'params': model.omics_adapter.parameters(), 'lr': 1e-5},
                {'params': model.wsi_attn_head.parameters(), 'lr': 2e-4},
                {'params': model.omics_attn_head.parameters(), 'lr': 1e-5},
                {'params': model.linear_head.parameters(), 'lr': 2e-4},
                {'params': model.classifier.parameters(), 'lr': 2e-4},
            ])
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
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
            for status in self.eval_set:
                values = self.results[status]
                score = self.validate(dataloaders['val'], model, criterion, status)
                if score > values["c-index"]:
                    values["c-index"] = score
                    values["checkpoint"] = model.state_dict()
                    values["epoch"] = epoch
                    self.save_checkpoint(state={"epoch": epoch, 
                                                "best_score": values["c-index"], 
                                                "state_dict": values["checkpoint"]},
                                                status=status)
                print(" *** best {} score={:.4f} at epoch {}".format(status, values["c-index"], values["epoch"]))
            scheduler.step()
            print('>>>')
            print('>>>')
        return self.results

    def train(self, data_loader, model, criterion, optimizer):
        model.train()

        total_loss = 0.0
        total_loss_multi, total_loss_wsi, total_loss_omics = 0.0, 0.0, 0.0
        cnt_wsi, cnt_omics = 0, 0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="Train Epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------Train Epoch {}-------------------------------".format(self.epoch))
        for batch_idx, (data_Study, data_ID, data_WSI, flag_WSI, data_Omics, flag_Omics, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            
            data_WSI = None if torch.sum(data_WSI).item() == 0 else data_WSI
            
            omics_inputs = dict()
            for i in range(len(data_Omics)):
                omics_inputs["x_omic%d" % (i + 1)] = data_Omics[i]
            if torch.cuda.is_available():
                for k, v in omics_inputs.items():
                    omics_inputs[k] = v.cuda()
            data_Omics = None if torch.sum(data_Omics[0]).item() == 0 else omics_inputs
            
            # prediction
            return_values = model(x_WSI=data_WSI, x_Omics=data_Omics, censor=data_Censorship, label=data_Label)

            loss = criterion(hazards=return_values["hazards"], S=return_values["S"], Y=data_Label, c=data_Censorship)
            loss_multi = loss.item()
            if data_WSI is None:
                loss_wsi = criterion(hazards=return_values["hazards_wsi"], S=return_values["S_wsi"], Y=data_Label, c=data_Censorship)
                loss += loss_wsi

            if data_Omics is None:
                loss_omics = criterion(hazards=return_values["hazards_omics"], S=return_values["S_omics"], Y=data_Label, c=data_Censorship)
                loss += loss_omics
            # results
            risk = -torch.sum(return_values["S"], dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            total_loss_multi += loss_multi
            if data_WSI is None:
                total_loss_wsi += loss_wsi.item()
                cnt_wsi += 1
            if data_Omics is None:
                total_loss_omics += loss_omics.item()
                cnt_omics += 1
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 20 == 0:
                print("batch: {}, loss: {:.4f}, loss_multi: {:.4f}".format(batch_idx, loss.item(), loss_multi))
                if data_WSI is None:
                    print("loss_wsi: {:.4f}".format(loss_wsi.item()))
                    
                if data_Omics is None:
                    print("loss_omics: {:.4f}".format(loss_omics.item()))
                    
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        loss_multi = total_loss_multi / len(dataloader)
        loss_wsi = total_loss_wsi / cnt_wsi if cnt_wsi > 0 else 0
        loss_omics = total_loss_omics / cnt_omics if cnt_omics > 0 else 0
        
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, loss_multi: {:.4f}, loss_wsi: {:.4f}, loss_omics: {:.4f}, c_index: {:.4f}'.format(loss, loss_multi, loss_wsi, loss_omics,
                                                                                                               c_index))

    def validate(self, data_loader, model, criterion, status="wsi-omics"):
        model.eval()
        total_loss = 0.0
        total_loss_multi, total_loss_wsi, total_loss_omics = 0.0, 0.0, 0.0
        cnt_wsi, cnt_omics = 0, 0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        all_attns = {}
        all_scores_wsi = {}
        all_scores_omics = {}
        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="Test Epoch {} with {}".format(self.epoch, status))
        else:
            dataloader = data_loader
            print("-------------------------------Test Epoch {} with {}-------------------------------".format(self.epoch, status))

        for batch_idx, (data_Study, data_ID, data_WSI, flag_WSI, data_Omics, flag_Omics, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            data_WSI = None if "wsi" not in status else data_WSI
            if "omics" in status:
                omics_inputs = dict()
                for i in range(len(data_Omics)):
                    omics_inputs["x_omic%d" % (i + 1)] = data_Omics[i]
                if torch.cuda.is_available():
                    for k, v in omics_inputs.items():
                        omics_inputs[k] = v.cuda()
                data_Omics = omics_inputs
            else:
                data_Omics = None
            
            with torch.no_grad():
                return_values = model(x_WSI=data_WSI, x_Omics=data_Omics, censor=data_Censorship, label=None)
            loss = criterion(hazards=return_values["hazards"], S=return_values["S"], Y=data_Label, c=data_Censorship)
            loss_multi = loss.item()
        
            if data_WSI is None:
                loss_wsi = criterion(hazards=return_values["hazards_wsi"], S=return_values["S_wsi"], Y=data_Label, c=data_Censorship)
                loss += loss_wsi

            if data_Omics is None:
                loss_omics = criterion(hazards=return_values["hazards_omics"], S=return_values["S_omics"], Y=data_Label, c=data_Censorship)
                loss += loss_omics
            # results
            attns = return_values["attns"]
            # stack attns on the first dimension
            attns = attns[-1].detach().cpu().numpy()
            all_attns[data_ID[0]] = attns
            if return_values["scores_path"] is not None:
                all_scores_wsi[data_ID[0]] = return_values["scores_path"].detach().cpu().numpy()
            if return_values["scores_omics"] is not None:
                all_scores_omics[data_ID[0]] = return_values["scores_omics"].detach().cpu().numpy()

            risk = -torch.sum(return_values["S"], dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            total_loss_multi += loss_multi
            if data_WSI is None:
                total_loss_wsi += loss_wsi.item()
                cnt_wsi += 1
            if data_Omics is None:
                total_loss_omics += loss_omics.item()
                cnt_omics += 1
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        loss_multi = total_loss_multi / len(dataloader)
        loss_wsi = total_loss_wsi / cnt_wsi if cnt_wsi > 0 else 0
        loss_omics = total_loss_omics / cnt_omics if cnt_omics > 0 else 0
        
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, loss_multi: {:.4f}, loss_wsi: {:.4f}, loss_omics: {:.4f}, c_index: {:.4f}'.format(loss, loss_multi, loss_wsi, loss_omics,
                                                                                                               c_index))
        return c_index

    def save_checkpoint(self, state, status):
        if status in self.filename_bests.keys():
            os.remove(self.filename_bests[status])
        filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          '{status}_model_best_{score:.4f}_{epoch}.pth.tar'.format(
                                              status=status, score=state['best_score'], epoch=state['epoch']))
        self.filename_bests[status] = filename_best
        print('save best [model] {filename} for {status}'.format(filename=filename_best,
                                                               status=status))
        os.makedirs(os.path.dirname(filename_best), exist_ok=True)
        torch.save(state, filename_best)
        
        
