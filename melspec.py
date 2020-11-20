
import numpy as np
import torch
import torch.utils.data

from scipy.io.wavfile import read
from stft import TacotronSTFT
import torchaudio
import librosa

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
        if loader == 'librosa': self.loader = librosa_load_file
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

        if sr != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sr, self.stft.sampling_rate))
        audio_norm = data.clamp(-1, 1)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm[None])
        melspec = torch.squeeze(melspec, 0)
        return melspec


# class WavglowMelLoader(torch.utils.data.Dataset):
#     """
#         1) loads audio,text pairs
#         2) normalizes text and converts them to sequences of one-hot vectors
#         3) computes mel-spectrograms from audio files.
        
#         Note: for VCTK, use scipy or librosa, for Librispeech use librosa, for VCC use scipy,
#         for commonvoice use torchaudio
#     """
#     def __init__(self, args, load_mel_from_disk=False, orig_freq=None, norm=False):
#         self.load_mel_from_disk = load_mel_from_disk
#         self.norm = norm
#         if not load_mel_from_disk:
#             self.max_wav_value = args.max_wav_value
#             self.sampling_rate = args.sampling_rate
#             self.stft = TacotronSTFT(
#                 args.filter_length, args.hop_length, args.win_length,
#                 args.n_mel_channels, args.sampling_rate, args.mel_fmin,
#                 args.mel_fmax)
#             self.resampler = torchaudio.transforms.Resample(orig_freq, new_freq=self.sampling_rate)

#     def get_mel(self, filename, raw_audio=None):
#         if not self.load_mel_from_disk:
#             if raw_audio is not None:
#                 audio = raw_audio
#             else:
#                 #audio, sampling_rate = load_wav_to_torch(filename)
                
#     #             audio, _falsesr = torchaudio.load(str(filename), normalization=True)
#                 audio, _ = librosa.load(str(filename), sr=self.sampling_rate)
#             if self.norm:
#                 audio = 0.95*(audio / abs(audio).max())
# #             t = pydub.AudioSegment.from_mp3(str(filename))
# #             audio = np.array(t.get_array_of_samples())/(2**15)
            
#             sampling_rate = self.sampling_rate
#             audio_norm = torch.from_numpy(audio[None]).float()
# #             audio_norm = self.resampler(audio)
            
#             #print(audio_norm.max(), audio_norm.min(), audio_norm.shape)
#             if sampling_rate != self.stft.sampling_rate:
#                 raise ValueError("{} {} SR doesn't match target {} SR".format(
#                     sampling_rate, self.stft.sampling_rate))
#             #audio_norm = audio / 32768.0
#             #audio_norm = audio_norm.unsqueeze(0)
#             audio_norm = audio_norm.clamp(-1, 1)
#             audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
#             melspec = self.stft.mel_spectrogram(audio_norm)
#             melspec = torch.squeeze(melspec, 0)
#         else:
#             melspec = torch.load(filename)
#             # assert melspec.size(0) == self.stft.n_mel_channels, (
#             #     'Mel dimension mismatch: given {}, expected {}'.format(
#             #         melspec.size(0), self.stft.n_mel_channels))

#         return melspec
