from typing import Tuple
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import random
import pandas as pd
import librosa
import numpy as np

class UtteranceDS(Dataset):

    def __init__(self, df: pd.DataFrame, sr, n_uttr_per_spk) -> None:
        super().__init__()
        self.df = df
        self.sr = sr
        self.n_uttr_per_spk = n_uttr_per_spk
        self.speakers = self.df.speaker.unique()

    def __len__(self): return len(self.speakers)

    def __getitem__(self, index) -> Tuple[Tensor, str]:
        options = self.df.loc[self.df.speaker == self.speakers[index], 'path']
        wav_paths = np.random.choice(options, self.n_uttr_per_spk, replace=False)
        wavs = [librosa.load(w, sr=self.sr)[0] for w in wav_paths]
        wavs = [torch.from_numpy(wav) for wav in wavs]

        return wavs

class SpecialCollater():

    def __init__(self, min_len, max_len) -> None:
        self.min_len = min_len
        self.max_len = max_len
        
    def create_batch(self, xs):
        batch_len = random.randint(self.min_len, self.max_len)
    
        xb = torch.zeros(len(xs), len(xs[0]), batch_len) 
        lb = torch.zeros(len(xs), len(xs[0]), dtype=torch.int)
        for i in range(len(xs)):
            for j in range(len(xs[0])):
                n_sam = xs[i][j].shape[0]
                if n_sam < batch_len:
                    xb[i, j, :n_sam] = xs[i][j]
                    lb[i, j] = n_sam
                else:
                    sp = random.randint(0, n_sam - batch_len)
                    xb[i, j] = xs[i][j][sp:sp+batch_len]
                    lb[i, j] = batch_len
        
        return xb, lb 

    def __call__(self, xs) -> Tuple[Tensor, Tensor]:
        return self.create_batch(xs)