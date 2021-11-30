from networks.speaker_embedder import SpeakerEmbedderGRU
from melspec import MelspecTransform
import torch
import torch.nn as nn
from hparams import hp

class GRUEmbedder(nn.Module):
    
    def __init__(self, normalize, **kwargs):
        super().__init__()
        self.model = SpeakerEmbedderGRU()
        self.hparams = hp.audio_waveglow
        self.melspec_tfm = MelspecTransform(self.hparams, **kwargs)
        self.should_normalize = normalize
    
    def forward(self, x):
        """ Takes in a set of mel spectrograms in shape (batch, frames, n_mels) """
        if self.should_normalize: x = self.normalize(x)
        return self.model(x)
    
    def normalize(self, x):
        _normer = -self.hparams.min_log_value/2
        return (x + _normer)/_normer

    def melspec_from_file(self, x):
        """ Convert file path `x` to a [n_frames, n_mels] shape mel-spectrogram tensor """
        return self.melspec_tfm.from_file(x).T
    
    def melspec_from_array(self, x, sr):
        """ Convert a 1D torch tensor in to a [n_frames, n_mels] shape mel-spectrogram tensor """
        return self.melspec_tfm.from_array(x, sr).T

    def print_hparams(self):
        for key in self.hparams.__dict__():
            if str(key).startswith('__') == True: continue
            print(key, ':', getattr(self.hparams, key))

def gru_embedder(pretrained=True, progress=True, normalize=True, **kwargs):
    r""" 
    GRU embedding model trained on the VCTK, CommonVoice, Librispeech, and VCC datasets.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        normalize (bool): whether the model should by default normalize input mel-spectrograms (default True)
        kwargs: arguments passed to the spectrogram transform
    """
    model = GRUEmbedder(normalize=normalize, **kwargs)
    if pretrained:
        state = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speaker-embedding/releases/download/0.1/gru-wrapped-f1f850.pth", 
                                                progress=progress)
        model.load_state_dict(state)

    return model
