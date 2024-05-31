"""Base dataset classes."""
import logging
import math
import random
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

BG_NOISE_PATH = "../Sample"
LOGGER = logging.getLogger(__name__)

SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480_000  # <- originally 64_600


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


class SimpleAudioFakeDataset(Dataset):
    def __init__(
        self,
        subset,
        augmentation=False,
        transform=None,
        return_label: bool = True,
        return_meta: bool = False,
    ):
        self.transform = transform
        self.samples = pd.DataFrame()
        self.augmentation = augmentation
        self.subset = subset
        self.allowed_attacks = None
        self.partition_ratio = None
        self.seed = None
        self.return_label = return_label
        self.return_meta = return_meta

    def split_samples(self, samples_list):
        if isinstance(samples_list, pd.DataFrame):
            samples_list = samples_list.sort_values(by=list(samples_list.columns))
            samples_list = samples_list.sample(frac=1, random_state=self.seed)
        else:
            samples_list = sorted(samples_list)
            random.seed(self.seed)
            random.shuffle(samples_list)

        p, s = self.partition_ratio
        subsets = np.split(
            samples_list, [int(p * len(samples_list)), int((p + s) * len(samples_list))]
        )
        return dict(zip(["train", "test", "val"], subsets))[self.subset]

    def df2tuples(self):
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append(
                (str(elem["path"]), elem["label"], elem["attack_type"])
            )

        self.samples = tuple_samples
        return self.samples

    def __getitem__(self, index) -> T_co:
        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]

            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
            if type(attack_type) != str and math.isnan(attack_type):
                attack_type = "N/A"
        else:
            path, label, attack_type = self.samples[index]
        if not self.augmentation:
            waveform, sample_rate = torchaudio.load(path, normalize=APPLY_NORMALIZATION)
            real_sec_length = len(waveform[0]) / sample_rate
    
            waveform, sample_rate = apply_preprocessing(waveform, sample_rate)
    
            return_data = [waveform, sample_rate]
            if self.return_label:
                label = 1 if label == "bonafide" else 0
                return_data.append(label)
    
            if self.return_meta:
                return_data.append(
                    (
                        attack_type,
                        path,
                        self.subset,
                        real_sec_length,
                    )
                )
            return return_data
        else:
            waveform, sample_rate = torchaudio.load(path, normalize=True)
            real_sec_length = len(waveform[0]) / sample_rate
    
            waveform, sample_rate = apply_preprocessing(waveform, sample_rate)
            n = random.random()
            if n < 0.2:
                waveform = add_noise(waveform)
            elif n < 0.4:
                waveform = add_gaussian_noise(waveform)
            elif n < 0.6:
                other_index = random.randint(0, len(self.sample) - 1)
                other_sample = self.sample.iloc[other_index] if isinstance(self.sample, pd.DataFrame) else self.sample[other_index]
                other_path = str(other_sample["path"])
                other_waveform, osr = torchaudio.load(other_path, normalize=True)
                other_waveform, osr = apply_preprocessing(other_waveform, osr)
    
                waveform = (waveform + other_waveform * 0.1) / 2
            elif n < 0.8:
                waveform = add_sound(waveform, index)

            return_data = [waveform, sample_rate]
            if self.return_label:
                label = 1 if label == "bonafide" else 0
                return_data.append(label)
            return return_data
    def __len__(self):
        return len(self.samples)


def apply_preprocessing(
    waveform,
    sample_rate,
):
    if sample_rate != SAMPLING_RATE and SAMPLING_RATE != -1:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    # Stereo to mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    # Trim too long utterances...
    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    # ... or pad too short ones.
    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def resample_file(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform


def add_noise(waveform, noise_level=0.001):
    noise = torch.FloatTensor(waveform.size()).uniform_(-1, 1) * noise_level
    noisy_waveform = waveform + noise
    return noisy_waveform

def add_gaussian_noise(signal, noise_level=0.005):
  noise = torch.randn_like(signal) * noise_level
  return signal + noise

def search_wav(directory):
  wav_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(".wav"):
        wav_files.append(os.path.join(root, file))
  return wav_files

wav_files = search_wav(BG_NOISE_PATH)

def add_sound(waveform, index, level=0.1):
    global wav_files
    random.seed(index)
    sound_file = random.choice(wav_files)
    sound, sample_rate = torchaudio.load(sound_file, normalize=True)

    # Stereo to mono
    if sound.size(0) > 1:
        sound = sound.mean(dim=0, keepdim=True)

    # Ensure waveform is 2D (batch x length)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Ensure sound is 2D
    if sound.dim() == 1:
        sound = sound.unsqueeze(0)

    # Adjust sound length to match waveform length
    if waveform.size(1) > sound.size(1):
        pad_size = waveform.size(1) - sound.size(1)
        sound = torch.nn.functional.pad(sound, (0, pad_size))
    else:
        sound = sound[:, :waveform.size(1)]

    # Add sound to waveform
    waveform = waveform + sound * level

    return waveform.squeeze(0)
