# Simple speaker embeddings
A speaker embedding network in Pytorch that is very quick to set up and use for whatever purposes.

## What?
A Pytorch model that takes in a log Mel-scale spectrogram and returns a 256-dimensional real vector of unit length known as an _embedding_ for the input speaker.
The vector is trained to be unique to the speaker identity of the input utterance -- so the returned vector should remain the same regardless of _what_ words are spoken in the input utterance, and depend only on the _speaker identity_ of who is speaking in the input utterance.

For example, if an input utterance saying "The quick brown fox" spoken by **speaker A** is fed into the model, the resulting 256-dimensional embedding should be close (in terms of Euclidean distance) to the embedding of an utterance saying "I like speaker embeddings" also spoken by **speaker A**. 
Conversely, the embedding should be far away (in terms of Euclidean distance) from the embedding of an utterance saying the same "The quick brown fox" spoken by **speaker B**.
Thus the embedding should be unique to the identity of the speaker of an input utterance, and not the linguistic content of the input utterance.
This is the behavior provided by the models in this repo.

**Quick info**:
- The input utterance can be of arbitrary length, although fairly short (5-10s) work best.
- For a most robust embedding generate several embedding vectors from multiple utterances/recordings and then take the mean of these vectors. Finally, scale the mean vector to again be of unit length. 
- The pretrained model works well on _unseen_ speakers. In other words, it generates reasonable and well-behaved embeddings for utterances by speakers never seen during training.
- The model is trained with the [GE2E loss](https://arxiv.org/abs/1710.10467) introduced for speaker verification, using the loss function [implementation provided by HarryVolek](https://arxiv.org/abs/1710.10467)
- The pretrained model currently available is a 3-layer GRU model with 768 hidden units in each layer. 
- The log Mel-scale spectrogram transform used is based on the ones used by Tacotron, and the transform functionality is bundled with the model in pytorch hub.

# Quick start
No cloning repos or downloading notebooks needed! Simply:
1. Ensure you have `pytorch`, `torchaudio`, `librosa`, `numpy`, and `scipy` python packages installed. 
2. Run: 
   ```python
    import torch
    model = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder')
    model.eval()
   ```
   Feed in `(batch size, frames, 80)` log Mel-spectrogram tensors and obtain `(batch size, 256)` embedding vectors. Trivial!

## Example
Lets go through a full example to show whats cracking.
First, assume you have an audio file `example.wav` ([like the `example.wav` in this repository](https://github.com/RF5/simple-speaker-embedding/raw/master/example.wav)) that is spoken by a speaker you would like an embedding of.
So, first step is to create the model and load the weights:

```python
import torch
model = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder')
model.eval()
```
This loads the pretrained model as a standard pytorch module. If you would like the untrained model, simply specify `pretrained=False` in the arguments for `torch.hub.load`. 

Now, we load the waveform to a log Mel-scale spectrogram:
```python
mel = model.melspec_from_file('example.wav')
```
Alternatively, if you already have a loaded waveform as a 1D Pytorch float tensor between -1 and 1, then you can also use `mel = model.melspec_from_array(x)` for 1D torch vector `x`. 

Finally, we can get the embedding by running the spectrogram through the model:
```python
embedding = model(mel[None]) # include [None] to add the batch dimension
```
`embedding` is now a rank 1, 256-dimensional tensor of unit length corresponding to the identity of the input speaker.

# Does it work?
Yes, to a pretty good extent. For example, below we plot the 2D t-SNE projection of several embeddings from utterances spoken by several speakers unseen during the training of the model. Each embedding is colored by the speaker identity of the corresponding utterance.

![cool picture](tsne-embedding.svg)

As is seen, utterances spoken by the same speaker are closely clustered, while utterances by different speakers remain well separated. This indicates good embedding behavior.

## Failure modes
The model is not perfect and fails under some conditions:
- If the input utterance contains multiple people talking, it will not generate sensible embeddings.
- If the input utterance is very long (>1min), then the model also becomes rather unstable.
- If the input utterance contains long portions of silence then the resulting embeddings lose meaning. 

# Hyperparameters

The hyperparameters of the model mainly include choices of actions involving the mel-spectrogram transform. To view them, simply go `model.print_hparams()`. It will return something like:
```python
filter_length  :  1024
hop_length  :  256
win_length  :  1024
n_mel_channels  :  80
sampling_rate  :  22050
mel_fmin  :  0
mel_fmax  :  8000.0
min_log_value  :  -11.52
```
All this information pertains to the settings of the Tacotron Mel-spectrogarm transform, except for the `min_log_value`, which is used by the model to normalize any input log Mel-scale spectrograms.

When using `melspec_from_file('example.wav')`, `example.wav` is automatically resampled to the correct sampling rate. However, if obtaining the mel-spectrogram from a torch array, please remember to resample the utterance first to 22050Hz. 

# Training
The model is trained on the combined VCC 2018, VCTK, Librispeech, and CommonVoice English datasets, using Fastai. 

To train the model:
1. Download these datasets
2. Precompute the log Mel-scale spectrograms for every utterance in each dataset
3. Save these precomputed spectrograms in a different folder for each dataset, with subfolders for each speaker.
4. Run the code in the training notebook `training.ipynb`

