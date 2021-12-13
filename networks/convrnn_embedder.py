import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from dataclasses import dataclass, field

def fix(blah): return field(default_factory=lambda: blah)

@dataclass
class ConvRNNConfig:
    conv_cfg: List[int] = fix([(512,10,5)] 
                                + [(512,3,2)] * 4 
                                + [(512,2,2)] * 2)

    conv_reduction_mult: int = 320
    conv_magic_offset: int = 400

    # Main RNN encoding parameters
    n_layers: int = 3
    n_hid: int = 768
    n_conv: int = 512
    fc_dim: int = 256
    hidden_p: float = 0.3
    bidir: bool = True


class ConvEncoder(nn.Module):
    def __init__(
        self, conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0, conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else: return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            self.conv_layers.append(
                block(in_d, dim, k, stride, 
                    is_group_norm = i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1) # --> (bs, seq_len, dim)
        return x

class ConvRNNEmbedder(nn.Module):

    def __init__(self, cfg:ConvRNNConfig, mode='train'): #conv_cfg, n_hid=768, n_conv=512, n_layers=3, fc_dim=256, hidden_p=0.3, bidir=False):
        super().__init__()    
        self.rnn_dim = cfg.n_hid
        self.conv_encoder = ConvEncoder(cfg.conv_cfg)

        self.rnn = nn.GRU(cfg.n_conv, cfg.n_hid, num_layers=cfg.n_layers, 
                            batch_first=True, dropout=cfg.hidden_p, bidirectional=cfg.bidir)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name: nn.init.constant_(param, 0.0)
            elif 'weight' in name: nn.init.xavier_normal_(param)
        self.head = nn.Linear(cfg.n_hid*(2 if cfg.bidir else 1), cfg.fc_dim)
        self.cfg = cfg
        self.mode = mode

    def forward(self, x, input_lengths):
        """ 
        Takes in a set of waveforms of shape (batch, uttrs_per_speaker, samples) 
        and associated waveform lengths (batch, uttrs_per_speaker)
        """

        bs, n_uttr_per_spk, T = x.shape
        x = x.view(bs*n_uttr_per_spk, T)
        input_lengths = input_lengths.view(bs*n_uttr_per_spk)
        if self.mode == 'train':
            perm = torch.randperm(bs*n_uttr_per_spk)
            unperm = torch.zeros_like(perm)
            for i,j in enumerate(perm): unperm[j] = i
            x = x[perm]
            input_lengths = input_lengths[perm]
        x = self.conv_encoder(x)
        lengths = (1 + (input_lengths - self.cfg.conv_magic_offset)/self.cfg.conv_reduction_mult).floor().long() 
        lengths = lengths.cpu()
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=total_length) # (bs, seq len, encoder embed dim)
        
        out_forward = outputs[range(len(outputs)), lengths - 1, :self.rnn_dim]
        out_reverse = outputs[:, 0, self.rnn_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), dim=1)
        pred = self.head(out_reduced) # (bs, n_speaker_logits)
        pred = pred / torch.norm(pred, p=2, dim=-1, keepdim=True)

        if self.mode == 'train':
            pred = pred[unperm]
            lengths = lengths[unperm]
        pred = pred.view(bs, n_uttr_per_spk, self.cfg.fc_dim)
        return pred

