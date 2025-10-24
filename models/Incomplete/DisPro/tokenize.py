import torch.nn as nn


class TokenizeWSI(nn.Module):
    def __init__(self, dim=256, n_features=1024) -> None:
        super(TokenizeWSI, self).__init__()
        self.WSI_net = nn.Sequential(nn.Linear(n_features, 512), 
                                     nn.ReLU(), 
                                     nn.Linear(512, dim), 
                                     nn.ReLU())
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x_WSI):
        tokens = self.WSI_net(x_WSI)
        # print('WSI tokens', tokens.shape)
        return tokens

class TokenizeOmics(nn.Module):
    def __init__(self, dim=256, n_features=1024) -> None:
        super(TokenizeOmics, self).__init__()
        self.tokenizer_net = nn.Sequential(nn.Linear(n_features, 512), 
                                     nn.ReLU(), 
                                     nn.Linear(512, dim), 
                                     nn.ReLU())
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, Omics):
        return self.tokenizer_net(Omics)