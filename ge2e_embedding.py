from networks.speaker_embedder import SpeakerEmbedderGRU
from melspec import MelspecTransform
import torch
import torch.nn as nn
from hparams import hp

class GRUEmbedder(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SpeakerEmbedderGRU()
        self.hparams = hp.audio_waveglow
        self.melspec_tfm = MelspecTransform(self.hparams, **kwargs)
    
    def forward(self, x):
        """ Takes in a set of mel spectrograms in shape (batch, frames, n_mels) """
        return self.model(x)

    def melspec_from_file(self, x):
        """ Convert file path `x` to a [n_mels, n_frames] shape mel-spectrogram tensor """
        return self.melspec_tfm.from_file(x)
    
    def melspec_from_array(self, x):
        """ Convert a 1D torch tensor in to a [n_mels, n_frames] shape mel-spectrogram tensor """
        return self.melspec_tfm.from_array(x)

def gru_embedder(pretrained=True, progress=True, **kwargs):
    r""" 
    GRU embedding model trained on the VCTK, CommonVoice, Librispeech, and VCC datasets.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        kwargs: arguments passed to the spectrogram transform
    """
    
    model = GRUEmbedder(**kwargs)
    if pretrained:

        state = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speaker-embedding/releases/download/0.1/gru-big-d0a8e22e.pth", 
                                                progress=progress)
        # state = torch.load('weights/resnet34.pth')
        model.load_state_dict(state)

    return model
