import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse


def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['train-clean-100', 'train-clean-360', 'train-other-500']
    for f in folders:
        all_files.extend(list((root_path/f).rglob('**/*.flac')))
    speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def make_vctk_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['wav48_silence_trimmed']
    for f in folders:
        all_files.extend(list((root_path/f).rglob('**/*.flac')))
    speakers = ['vctk-' + f.parent.stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def make_commonvoice_df(root_path: Path, min_uttr_per_spk=30, max_uttr_per_spk=500) -> pd.DataFrame:
    bdf = pd.read_csv(root_path/'validated.tsv', sep='\t')
    df = bdf[['path', 'client_id']]
    df = df.rename({'client_id': 'speaker'}, axis=1)
    df['cnt'] = df.groupby('speaker')['path'].transform('count')
    df = df[df.cnt > min_uttr_per_spk]
    df = df[df.cnt < max_uttr_per_spk]
    df.drop('cnt', axis=1)
    df['speaker'] = 'cv-' + df['speaker']
    return df

def make_vox1_df(root_path: Path) -> pd.DataFrame:
    all_files = list((root_path/'wav').rglob('**/*.wav'))
    speakers = ['vox1-' + f.parents[1].stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def make_vox2_df(root_path: Path) -> pd.DataFrame:
    all_files = list((root_path/'dev/wav').rglob('**/*.wav'))
    speakers = ['vox2-' + f.parents[1].stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate train & valid csvs from dataset directories")

    parser.add_argument('--librispeech_path', default=None, type=str)
    parser.add_argument('--vctk_path', default=None, type=str)
    parser.add_argument('--commonvoice_path', default=None, type=str)
    parser.add_argument('--vox1_path', default=None, type=str)
    parser.add_argument('--vox2_path', default=None, type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--valid_spks', default=200, type=int)

    args = parser.parse_args()

    cat_dfs = []
    if args.librispeech_path is not None:
        ls_df = make_librispeech_df(Path(args.librispeech_path))
        cat_dfs.append(ls_df)
    if args.vctk_path is not None:
        vctk_df = make_vctk_df(Path(args.vctk_path))
        cat_dfs.append(vctk_df)
    if args.commonvoice_path is not None:
        cv_df = make_commonvoice_df(Path(args.commonvoice_path))
        cat_dfs.append(cv_df)
    if args.vox1_path is not None:
        vox1_df = make_vox1_df(Path(args.vox1_path))
        cat_dfs.append(vox1_df)
    if args.vox2_path is not None:
        vox2_df = make_vox2_df(Path(args.vox2_path))
        cat_dfs.append(vox2_df)

    full_df = pd.concat(cat_dfs)
    print("Preliminary number of speakers: ", len(full_df.speaker.unique()))
    full_df['cnt'] = full_df.groupby('speaker')['path'].transform('count')
    full_df = full_df[full_df.cnt > 8] # Trim to only speakers with at least 8 utterances
    full_df.drop('cnt', axis=1)

    speakers = full_df.speaker.unique()
    np.random.seed(args.seed)
    np.random.shuffle(speakers)
    n_valid = args.valid_spks
    train_spks = speakers[:-n_valid]
    valid_spks = speakers[-n_valid:]
    train_df = full_df[full_df.speaker.isin(train_spks)]
    print(f"Finished constructing train df of {len(train_df):,d} utterances.")
    valid_df = full_df[full_df.speaker.isin(valid_spks)]
    print(f"Finished constructing valid df of {len(valid_df):,d} utterances ({len(valid_df.speaker.unique())} speakers).")
    os.makedirs('splits', exist_ok=True)
    train_df.to_csv("splits/train.csv.zip", index=False, compression='zip')
    valid_df.to_csv("splits/valid.csv.zip", index=False, compression='zip')

if __name__ == '__main__':
    main()
