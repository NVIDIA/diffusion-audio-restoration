# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import os
import librosa
import torch
import csv
import numpy as np
from torchaudio import functional as F
from typing import List, TypeVar
from audio_transforms.transforms import apply_audio_transforms
from math import floor
from corruption.corruptions import UpsampleMask as UpsampleMask

T = TypeVar("T")

def read_maestro_csv(root_folder, filename):
    all_files = {
        "train": [],
        "validation": [],
        "test": []
    }

    with open(os.path.join(root_folder, filename)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            assert len(row) == 7
            composer, title, split, year, midi_filename, audio_filename, duration = row
            split = split.strip()
            audio_filename = audio_filename.strip()
            duration = float(duration)
            sample_rate = 44100 if int(year) <= 2015 else 48000
            all_files[split].append((os.path.join(root_folder, audio_filename), duration, sample_rate))

    return all_files

def read_standard_csv(root_folder, filename, max_sr=44100, apply_sr_loss_mask=False):
    all_files = {
        "train": [],
        "validation": [],
        "test": []
    }

    with open(os.path.join(root_folder, filename)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            estimated_true_sr = max_sr
            if len(row) == 3:
                split, audio_filename, duration = row
            elif len(row) == 4:
                split, audio_filename, duration, estimated_true_sr = row
                if apply_sr_loss_mask == False:
                    estimated_true_sr = max_sr
                else:
                    if int(estimated_true_sr) < 32000:
                        # avoid all-zero masks when cutoff is sampled to be max (16khz)
                        continue
            split = split.strip()
            audio_filename = audio_filename.strip()
            duration = float(duration)
            sample_rate = int(estimated_true_sr)
            all_files[split].append((audio_filename, duration, sample_rate))

    return all_files

class MixAudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 mix_dataset_config={},
                 split='train',
                 segment_length=2**16,
                 sampling_rate=44100,
                 max_samples=None,
                 transforms_gt=[],
                 transforms_aug=[],
                 eval_transforms_aug=[],
                 evaluation_mode=False,
                 # opt, log, corrupt_method
                 ):
        super(MixAudioDataset, self).__init__()

        self.mix_dataset_config = mix_dataset_config
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

        assert split in ["train", "validation", "test"]
        print('reading csv')

        self.all_files = {
            "train": [],
            "validation": [],
            "test": []
        }
        for dataset_name in mix_dataset_config:
            root_folder = mix_dataset_config[dataset_name]['root_folder']
            filename = mix_dataset_config[dataset_name]['filename']
            apply_sr_loss_mask = False
            if 'apply_sr_loss_mask' in mix_dataset_config[dataset_name].keys():
                apply_sr_loss_mask = mix_dataset_config[dataset_name]['apply_sr_loss_mask']
            all_files = read_standard_csv(root_folder, filename, sampling_rate, apply_sr_loss_mask=apply_sr_loss_mask)
            print("{} has {} training files".format(dataset_name, len(all_files['train'])))

            all_files['validation'] = all_files['validation'][:100]  # faster validation compute

            for key in self.all_files.keys():
                self.all_files[key] += all_files[key]

        self.split_files = self.all_files[split]
        self.transforms_gt = transforms_gt
        self.transforms_aug = transforms_aug
        self.split = split
        self.evaluation_mode = evaluation_mode
        if evaluation_mode:
            self.transforms_aug = eval_transforms_aug

        print('building file <-> index mapping')
        self.mapped_list = self.build_file_idx_mapping()
        if max_samples is not None:
            self.mapped_list = self.mapped_list[:max_samples]

        print("Loaded {} samples for {} split".format(len(self.mapped_list), split))

    def build_file_idx_mapping(self, verbose=False):
        # map between sample index and each audio based on segment_length and their duration
        # [(0 {which means first sample}, 0.0, 2.97), (0, 2.97, 5.94), ..., (n, start, end)]
        mapped_list = []
        for sample_idx, sample in enumerate(self.split_files):
            audio_filename, duration, sample_rate = sample
            segment_time = float(self.segment_length) / float(self.sampling_rate) + 0.001
            n_segments = int(np.floor(duration / segment_time))
            for i in range(n_segments):
                mapped_list.append((sample_idx, i * segment_time, (i + 1) * segment_time))
        return mapped_list

    def __len__(self):
        return len(self.mapped_list)

    def load_wav_to_torch(self, audiopath, start_time=None, end_time=None):
        audio, sr = librosa.load(audiopath, sr=None)
        if len(audio.shape) != 1:
            audio = librosa.to_mono(audio.T)  # (L, 2) -> (2, L) -> mono-channel
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)

        crop_start = floor(start_time * self.sampling_rate)
        crop_start = max(0, min(crop_start, len(audio) - self.segment_length))

        audio = audio[crop_start:crop_start+self.segment_length]
        if len(audio) < self.segment_length:
            audio = np.append(audio, [0] * (self.segment_length - len(audio)))

        audio = torch.from_numpy(audio).float()
        # FIXME: might be good to add better normalization here
        if audio.max() > 1 or audio.min() < -1:
            audio = audio / audio.abs().max()
        if audio.min() > 0:
            audio = 2 * audio - 1

        return audio

    def unstable_getitem(self, index):
        sample_idx, start_time, end_time = self.mapped_list[index]
        # audio_sample_rate is the sampling rate estimated for this specific sample
        audio_filename, duration, audio_sample_rate = self.split_files[sample_idx]

        audio = self.load_wav_to_torch(audio_filename, start_time=start_time, end_time=end_time)
        stft_target, _ = apply_audio_transforms(audio, self.transforms_gt)

        # stft_transformed, mask = apply_audio_transforms(audio, self.transforms_aug)
        stft_transformed, mask = apply_audio_transforms(stft_target, self.transforms_aug)

        # now get the loss mask, which is inverse of the upsampling mask
        # we'll use this to mask the loss for upper frequency regions not present in the original audio
        # mask up to frequency corresponding to 1/2 sampling rate

        true_bandwidth_mask = 1-UpsampleMask.get_upsample_mask(stft_target, audio_sample_rate//2, audio_sample_rate//2, sampling_rate=self.sampling_rate, dc_dropped=True)
        mask = mask * true_bandwidth_mask

        return {"x_0_clean": stft_target,
                "x_0_corrupted": stft_transformed,
                "loss_mask": mask,
                "original_audio_sample_rate": audio_sample_rate,
                "seq_lens": stft_target.shape[-1],
                "x_0_wav": audio
                }

    def __getitem__(self, index):
        try:
            dic = self.unstable_getitem(index)
        except Exception as e:
            print('sample {} cannot be loaded due to {}'.format(index, e))
            dic = self.unstable_getitem((index+42)%99)

        return dic


class FullSequencePredictDataset(torch.utils.data.Dataset):
    """
    Applies prediction for full audio sequence segmenting
    Make sure used only with batch size 1
    """
    def __init__(self, audio_file_list,
                 sampling_rate=44100,
                 transforms_gt=[],
                 transforms_aug=[],
                 start_idx=0,
                 end_idx=None
                 ):
        super().__init__()
        self.transforms_gt = transforms_gt
        self.transforms_aug = transforms_aug
        if end_idx is None:
            end_idx = len(audio_file_list)
        self.audio_file_list = audio_file_list[start_idx:end_idx]
        print("predicting filelist elements %d through %d" %(start_idx, end_idx-1))
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, index):
        filepath = self.audio_file_list[index]['filepath']
        outdir = self.audio_file_list[index]['output_subdir']

        audio, _ = librosa.load(filepath, sr=self.sampling_rate)
        if len(audio.shape) != 1:
            audio = librosa.to_mono(audio.T)  # (L, 2) -> (2, L) -> mono-channel
        audio = torch.from_numpy(audio)
        stft_target, _ = apply_audio_transforms(audio, self.transforms_gt)

        stft_transformed, mask = apply_audio_transforms(stft_target, self.transforms_aug)

        return {"x_0_clean": stft_target,
                "x_0_corrupted": stft_transformed,
                "loss_mask": mask,
                "x_0_wav": audio,
                "outdir": outdir,
                "output_sr": self.sampling_rate
                }
