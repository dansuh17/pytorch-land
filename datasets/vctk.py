"""
VCTK clean-noisy speech dataset.

download from: https://datashare.is.ed.ac.uk/handle/10283/2791

If you unzip the file, it inflates to:
```
Archive:  DS_10283_2791.zip
  inflating: license_text
  inflating: clean_testset_wav.zip
  inflating: clean_trainset_28spk_wav.zip
  inflating: clean_trainset_56spk_wav.zip
  inflating: logfiles.zip
  inflating: noisy_testset_wav.zip
  inflating: noisy_trainset_28spk_wav.zip
  inflating: noisy_trainset_56spk_wav.zip
  inflating: testset_txt.zip
  inflating: trainset_28spk_txt.zip
  inflating: trainset_56spk_txt.zip
```
"""
import re
import os
import random
import math
import scipy
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from tqdm import tqdm


N_MELS = 40  # number of mel filters
# basic info of vctk dataset used for unzipping
CLEAN_TESTSET_WAV = 'clean_testset_wav'
CLEAN_TRAINSET_28SPK_WAV = 'clean_trainset_28spk_wav'
CLEAN_TRAINSET_56SPK_WAV = 'clean_trainset_56spk_wav'
NOISY_TESTSET_WAV = 'noisy_testset_wav'
NOISY_TRAINSET_28SPK_WAV = 'noisy_trainset_28spk_wav'
NOISY_TRAINSET_56SPK_WAV = 'noisy_trainset_56spk_wav'
# target directory after unzipping
CLEAN_TRAINSET_DIR = 'clean_trainset'
NOISY_TRAINSET_DIR = 'noisy_trainset'


def load_vctk_dataloaders(data_path: str, batch_size: int):
    """
    Load VCTK dataset (in whatever form) dataloaders.

    Args:
        data_path (str): data path whtere datasets are located
        batch_size (int): size of batch

    Returns:
        train_dataloader, validate_dataloader, test_dataloader
    """
    # create datasets
    train_dataset = NoisyVCTKSpectrogram(data_path)
    validate_dataset = NoisyVCTKSpectrogram(data_path)
    test_dataset = NoisyVCTKSpectrogram(data_path)

    # calculate indices out of dataset size
    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = math.floor(num_data * 0.8)
    num_valtest = num_data - num_train
    num_validate = num_valtest // 2

    train_idx, valtest_idx = indices[:num_train], indices[num_train:]
    validate_idx, test_idx = valtest_idx[:num_validate], valtest_idx[num_validate:]

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler.SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size)
    validate_dataloader = DataLoader(
        validate_dataset,
        sampler=sampler.SubsetRandomSampler(validate_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=sampler.SubsetRandomSampler(test_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size)
    return train_dataloader, validate_dataloader, test_dataloader


def unpack_dataset(fname: str, out_path: str):
    """
    Unpacks VCTK dataset.

    Args:
        fname (str): the file name of zipped vctk
        out_path (str): output directory
    """
    os.system('unzip -d {} {}'.format(out_path, fname))
    # data output directory
    clean_dir = os.path.join(out_path, CLEAN_TRAINSET_DIR)
    os.makedirs(clean_dir, exist_ok=True)
    noisy_dir = os.path.join(out_path, NOISY_TRAINSET_DIR)
    os.makedirs(noisy_dir, exist_ok=True)

    # unzip clean audio files
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(CLEAN_TESTSET_WAV))))
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(CLEAN_TRAINSET_28SPK_WAV))))
    # move the datasets for 28 and 56 speakers in the same folder
    os.system('mv {}/* {}'.format(os.path.join(out_path, CLEAN_TRAINSET_28SPK_WAV), clean_dir))
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(CLEAN_TRAINSET_56SPK_WAV))))
    os.system('mv {}/* {}'.format(os.path.join(out_path, CLEAN_TRAINSET_56SPK_WAV), clean_dir))

    # unzip noisy audio files
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(NOISY_TESTSET_WAV))))
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(NOISY_TRAINSET_28SPK_WAV))))
    # move the datasets for 28 and 56 speakers in the same folder
    os.system('mv {}/* {}'.format(os.path.join(out_path, NOISY_TRAINSET_28SPK_WAV), noisy_dir))
    os.system('unzip -d {} {}'.format(
        out_path, os.path.join(out_path, '{}.zip'.format(NOISY_TRAINSET_56SPK_WAV))))
    os.system('mv {}/* {}'.format(os.path.join(out_path, NOISY_TRAINSET_56SPK_WAV), noisy_dir))


def noisy_vctk_preprocess(in_path: str, out_path: str, clean_dir: str, noisy_dir: str,
                          target_sr=16000, window_size=256, hop_size: int=None, split_size=11, mel=True):
    if hop_size is None:
        hop_size = window_size // 2  # 0.5 hop by default

    clean_data_path = os.path.join(in_path, clean_dir)
    print('Preprocessing : {} and {}'.format(clean_dir, noisy_dir))
    for fname in tqdm(os.listdir(clean_data_path), ascii=True):
        # parse the file name that should have form : 'p125_111.wav'
        re_match = re.match(r'p(?P<class_num>\d+)_(?P<wav_id>\d+)\.wav', fname)
        class_num = re_match.group('class_num')
        out_dir = os.path.join(out_path, class_num)
        os.makedirs(out_dir, exist_ok=True)
        wav_id = re_match.group('wav_id')
        full_path = os.path.join(in_path, clean_dir, fname)

        # also read the noisy audio as well
        noisy_counterpart_path = os.path.join(in_path, noisy_dir, fname)

        # read the audio file
        clean_y, _ = librosa.load(full_path, sr=target_sr)  # resample as it reads
        noisy_y, _ = librosa.load(noisy_counterpart_path, sr=target_sr)

        # create mel-spectrogram - TODO: deprecated option?
        if mel:
            # shape=(n_mels, t)
            clean_spec = librosa.feature.melspectrogram(
                clean_y, sr=target_sr, n_fft=window_size, hop_length=hop_size, n_mels=N_MELS)
            noisy_spec = librosa.feature.melspectrogram(
                noisy_y, sr=target_sr, n_fft=window_size, hop_length=hop_size, n_mels=N_MELS)

            # split the spectrogram by 'split_size'-frames-sized chunks
            clean_split = _split_spectrogram(clean_spec, chunk_size=split_size)
            noisy_split = _split_spectrogram(noisy_spec, chunk_size=split_size)

            # save each clean-noisy pairs of chunks to file
            for split_id, split_pair in enumerate(zip(clean_split, noisy_split)):
                out_fname = '{}_s{:04}'.format(wav_id, split_id)
                np.save(os.path.join(out_dir, out_fname),
                        np.asarray(split_pair))


def inverse_mel(mel_spec, sr: int, n_fft: int):
    """
    Inverse mel-spectrum

    Args:
        mel_spec: mel spectrogram
        sr (int): sample rate
        n_fft (int): number of fft bins

    Returns:
        ys: signal as a result of inverse fft
    """
    # stitch together
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=N_MELS)
    power_spec = np.dot(np.matrix(mel_basis).I, mel_spec)
    _, ys = scipy.signal.istft(power_spec, fs=sr)
    return ys


def _split_spectrogram(spec, chunk_size: int):
    """
    Splits spectrogram into chunks that have sizes of
    'chunk_size' in the time axis.

    Args:
        spec (np.ndarray): spectrogram represented as 2D array
        chunk_size (int): number of frames to split into

    Returns:
        split_spec = list of np.ndarray's representing chunks having ``chunk_size`` sizes
    """
    time_length = spec.shape[1]
    num_chunks = time_length // chunk_size
    spec = spec[:, :num_chunks * chunk_size]
    return np.hsplit(spec, num_chunks)


class NoisyVCTKSpectrogram(Dataset):
    """
    Dataset for VCTK dataset that has been preprocessed into mel-spectrogram chunks.
    The dataset returns pairs of clean and noisy mel-spectrograms.
    """
    def __init__(self, data_path: str):
        super().__init__()
        data = []
        for speaker_id in os.listdir(data_path):
            speaker_data_dir = os.path.join(data_path, speaker_id)
            for fname in os.listdir(speaker_data_dir):
                # append data path into the list of all audio data
                data.append(os.path.join(speaker_data_dir, fname))
        self.data = data

    def __getitem__(self, idx):
        pair = np.load(self.data[idx])
        # return clean, noisy pair
        return pair[0], pair[1]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # run this script in order to preprocess the dataset
    path = os.path.join(os.path.dirname(__file__))
    dataset_path = os.path.join(path, 'vctk_corpus')
    unpack_dataset('DS_10283_2791.zip', out_path=dataset_path)

    out_path = 'vctk_processed'
    # preprocess train set
    noisy_vctk_preprocess(
        in_path=dataset_path,
        out_path=out_path,
        noisy_dir=NOISY_TRAINSET_DIR,
        clean_dir=CLEAN_TRAINSET_DIR)
    # preprocess test set
    noisy_vctk_preprocess(
        in_path=dataset_path,
        out_path=out_path,
        noisy_dir=NOISY_TESTSET_WAV,
        clean_dir=CLEAN_TESTSET_WAV)