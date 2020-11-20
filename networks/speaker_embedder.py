import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastai.text.all import RNNDropout, WeightDropout, Module, store_attr, one_param, to_detach        

class SpeakerEncoder2(nn.Module):
    
    def __init__(self, n_hid=768, n_mels=80, n_layers=2, fc_dim=256, hidden_p=0.3):
        super(SpeakerEncoder2, self).__init__()    
        self.rnn_stack = nn.GRU(n_mels, n_hid, num_layers=n_layers, batch_first=True, dropout=hidden_p)
        for name, param in self.rnn_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(n_hid, fc_dim)
        
    def forward(self, x):
        """ Takes in a set of mel spectrograms in shape (batch, frames, n_mels) """
        x, _ = self.rnn_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
