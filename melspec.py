
import librosa
import numpy as np
import torch
import torch.utils.data
import torchaudio
from scipy.io.wavfile import read

from stft import TacotronSTFT

def scipy_load_file(file_path):
    sr, data = read(file_path)
    return torch.FloatTensor(data.astype(np.float32)), sr

def librosa_load_file(file_path, target_sr=None):
    data, sr = librosa.load(str(file_path), sr=target_sr)
    return torch.FloatTensor(data.astype(np.float32)), sr

def torchaudio_load_file(file_path, normalization=True):
    data, sr = torchaudio.load(str(file_path))
    return data.float(), sr

class MelspecTransform():
    def __init__(self, transform_hparams, normalize=False, resample=None, loader='librosa'):
        """
        `trasnform_hparams`: config for the Tacotron STFT (e.g hop and window length, original sampling rate)
        `normalize`: whether to normalize the volume of an input audio array first to roughly between (-1, 1)
        `resample`: a target sampling rate to resample the audio to. `None` indicates no resampling.
        `loader`: the library used to load audio from files. Options are `librosa`, `scipy`, or `torchaudio`.
        """
        args = transform_hparams
        self.stft = TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.sampling_rate = args.sampling_rate
        self.normalize = normalize

        if resample: self.sampling_rate = int(resample)
        if loader == 'librosa': self.loader = lambda x: librosa_load_file(x, target_sr=self.sampling_rate)
        elif loader == 'scipy': self.loader = scipy_load_file
        elif loader == 'torchaudio': self.loader = torchaudio_load_file
        else: raise NotImplementedError("Loading method not implemented!")
    
    def from_file(self, path):
        """ Get mel spectrogram from an audio file """
        data, sr = self.loader(path)
        return self.from_array(data, sr)

    def from_array(self, data, sr=None):
        """ Get mel spectrogram from array of waveform magnitude samples """
        if self.normalize: data = 0.95*(data / abs(data).max())

        print(sr, self.stft.sampling_rate)
        if sr != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format( \
                sr, self.stft.sampling_rate))
        audio_norm = data.clamp(-1, 1)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm[None])
        melspec = torch.squeeze(melspec, 0)
        return melspec

