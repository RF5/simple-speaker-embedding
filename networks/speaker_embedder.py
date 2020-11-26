import torch
import torch.nn as nn

class SpeakerEmbedderGRU(nn.Module):

    def __init__(self, n_hid=768, n_mels=80, n_layers=3, fc_dim=256, hidden_p=0.3, bidir=False):
        super().__init__()    
        self.rnn_stack = nn.GRU(n_mels, n_hid, num_layers=n_layers, 
                            batch_first=True, dropout=hidden_p, bidirectional=bidir)
        for name, param in self.rnn_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(n_hid, fc_dim)
        
    def forward(self, x):
        """ Takes in a set of mel spectrograms in shape (batch, frames, n_mels) """
        self.rnn_stack.flatten_parameters()            
        
        x, _ = self.rnn_stack(x) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,-1,:]
        x = self.projection(x)
        x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return x
