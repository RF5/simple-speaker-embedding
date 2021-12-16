from typing import Dict
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import numpy as np
import logging
import random
import librosa
from torch import Tensor
from fastprogress.fastprogress import progress_bar

def eval(args):
    if args.model == 'gru_embedder':
        model = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').to(args.device)
        if args.checkpoint_override is not None:
            print(f"Loading override checkpoint from {args.checkpoint_override}")
            ckpt = torch.load(args.checkpoint_override, map_location=args.device)
            model.load_state_dict(ckpt)
    elif args.model == 'convgru_embedder':
        model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder').to(args.device)
        if args.checkpoint_override is not None:
            print(f"Loading override checkpoint from {args.checkpoint_override}")
            ckpt = torch.load(args.checkpoint_override, map_location=args.device)
            model.model.load_state_dict(ckpt['model_state_dict'])
    else: raise NotImplementedError()
    print(args)
    model = model.eval().to(args.device)
    print(f"Loaded pretrained model {model}")
    test_df = pd.read_csv(args.test_csv)
    print(f"Loaded test df of length {len(test_df)}, running eer with same p {args.eer_p_same} and seed {args.seed}.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    same_p = torch.ones(len(test_df)).fill_(args.eer_p_same)
    sames = torch.bernoulli(same_p).bool()

    cosims = []
    for i, row in progress_bar(test_df.iterrows(), total=len(test_df)):
        same = sames[i]
        if same:
            # find all same utterances
            _df = test_df[test_df.speaker == row.speaker]
            paths = list(_df.path)
            paths.remove(row.path)
        else: 
            # find all different speaker utterances
            _df = test_df[test_df.speaker != row.speaker]
            paths = list(_df.path)

        targ_path = random.choice(paths)
        
        with torch.no_grad():
            if args.model == 'gru_embedder':
                mel = model.melspec_from_file(row.path).to(args.device)
                mel2 = model.melspec_from_file(targ_path).to(args.device)
                x1_emb = model(mel[None])[0]
                x2_emb = model(mel2[None])[0]
            elif args.model == 'convgru_embedder':
                audio, _ = librosa.load(row.path, sr=16000)
                audio = torch.from_numpy(audio).float().to(args.device)

                audio2, _ = librosa.load(targ_path, sr=16000)
                audio2 = torch.from_numpy(audio2).float().to(args.device)

                x1_emb = model(audio[None])[0]
                x2_emb = model(audio2[None])[0]
            cosim = F.cosine_similarity(x1_emb, x2_emb, dim=-1).cpu()
        
        cosims.append(cosim)
    cosims = torch.stack(cosims, dim=0)
    eer, thresh = calculate_eer(sames, cosims)
    print((
        "| metric | value | \n"
        "| ------ | ----- | \n"
        f"| EER    | {eer:5.4f} |"
    ))
    print("\nNow plotting results as a UMAP figure for 8 speakers with 8 utterances each.")
    speakers = list(test_df.speaker.unique())
    speakers = random.sample(speakers, k=8)
    spk_dict = {}
    for s in progress_bar(speakers):
        _df = test_df[test_df.speaker == s]
        paths = random.sample(list(_df.path), k=8)
        embs = []
        for p in paths:
            with torch.no_grad():
                if args.model == 'gru_embedder':
                    mel = model.melspec_from_file(p).to(args.device)
                    x1_emb = model(mel[None])[0]
                elif args.model == 'convgru_embedder':
                    audio, _ = librosa.load(p, sr=16000)
                    audio = torch.from_numpy(audio).float().to(args.device)
                    x1_emb = model(audio[None])[0]
                embs.append(x1_emb.cpu())
        spk_dict[s] = torch.stack(embs, dim=0)
    print("Embeddings for umap gathered, computing transform.")
    project_umap(spk_dict, args.seed)

def project_umap(spk_dict: Dict[str,Tensor], seed):
    sorted_speakers = sorted(list(spk_dict.keys()))
    flat_embs = torch.cat([spk_dict[k] for k in sorted_speakers], dim=0).numpy()
    try:
        from umap import UMAP
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Please install umap, sklearn, and matplotlib from pypi to plot umap results.')
    data = StandardScaler().fit_transform(flat_embs)
    reducer = UMAP(metric='cosine', verbose=True, n_neighbors=20, random_state=seed)
    reduced_data = reducer.fit_transform(data)
    print(reduced_data.shape)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    reduced_chunks = torch.from_numpy(reduced_data).chunk(len(spk_dict), dim=0)
    for s, c in zip(sorted_speakers, reduced_chunks):
        ax.scatter(c.numpy()[:, 0], c.numpy()[:, 1])
    ax.legend(sorted_speakers)
    ax.set_xlabel('umap 1st component')
    ax.set_ylabel('umap 2nd component')
    ax.set_title("2D umap projection with n_neighbors=20")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('umap_plot.svg')
    print("Saved umap plot to umap_plot.svg")

def calculate_eer(y: Tensor, y_score: Tensor, pos=1):
    """ 
    Method to compute eer, retrieved from https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_EER.py 
    `y` is tensor of (cnt, ) of labels (0 or 1)
    `y_score` is tensor of (cnt, ) of similarity scores
    `pos` is the positive label, 99% of the time leave it as 1.
    """
    try:
        from scipy.interpolate import interp1d
        from scipy.optimize import brentq
        from sklearn.metrics import roc_curve
    except ModuleNotFoundError: 
        raise ModuleNotFoundError("Problem: for EER metrics, you require scipy and sklearn. Please install them first.")
    y = y.numpy()
    y_score = y_score.numpy()
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def main():
    print('Initializing evaluation process..')
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="script to evaluate EER and umap plots of a desired model")
    parser.add_argument('--model', type=str, choices=['convgru_embedder', 'gru_embedder'], required=True)
    parser.add_argument('--checkpoint_override', type=str, required=False, help='path to a checkpoint to load for the model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--eer_p_same', type=float, default=0.5, help="Probability that paired utterance is from same speaker.")
    parser.add_argument('--seed', type=int, default=1775)
    args = parser.parse_args()
    eval(args)

if __name__ == '__main__':
    main()