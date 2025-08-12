# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, List
from datasets.datasets import MixAudioDataset, FullSequencePredictDataset
from utils import SequenceLength
import numpy as np

def collate_fn(batch):
    all_lengths = torch.tensor([mel.shape[1] for mel in batch])

    sequence_lengths = SequenceLength(all_lengths)
    max_length = all_lengths.max()
    channels = batch[0].shape[0]
    # pad length to closest power of 2
    # we assume mel channels fixed at 80, which is divisible by 2 up to 3 times
    padded_length = int(2**np.ceil(np.log2(all_lengths.max())))
    all_mels = torch.zeros((len(batch), channels, padded_length))
    for ind in range(len(batch)):
        all_mels[ind, :, :all_lengths[ind]] = torch.tensor(batch[ind])

    output_dict = {'mels': all_mels, 'seq_lens': sequence_lengths}
    return output_dict


class STFTAudioDataModule(pl.LightningDataModule):
    def __init__(self,
                 mix_dataset_config={},
                 segment_length=2**16,
                 sampling_rate=22050,
                 num_workers=0,
                 batch_size=8,
                 transforms_gt=[],
                 transforms_aug=[],
                 transforms_aug_val=[],
                 eval_transforms_aug=[],
                 train_max_samples=None,
                 val_max_samples=100,
                 predict_filelist=[],
                 predict_start_idx=0,
                 predict_end_idx=None
                 ):
        super().__init__()
        # self.fft_size = fft_size
        # self.hop_size = hop_size
        # self.win_length = win_length
        # self.window_type = window_type
        self.mix_dataset_config = mix_dataset_config
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transforms_aug = transforms_aug
        if not transforms_aug_val:
            self.transforms_aug_val = transforms_aug
        else:
            self.transforms_aug_val = transforms_aug_val  # validation set augmentation (randomness is fixed)
        self.transforms_gt = transforms_gt
        self.eval_transforms_aug = eval_transforms_aug
        self.train_max_samples = train_max_samples
        self.val_max_samples = val_max_samples
        self.predict_filelist = predict_filelist
        self.predict_start_idx = predict_start_idx
        self.predict_end_idx = predict_end_idx


    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            print("initializing training dataset")
            self.trainset = MixAudioDataset(
                mix_dataset_config = self.mix_dataset_config,
                split = 'train',
                segment_length=self.segment_length,
                sampling_rate=self.sampling_rate,
                transforms_aug = self.transforms_aug,
                transforms_gt = self.transforms_gt,
                eval_transforms_aug = self.eval_transforms_aug,
                max_samples = self.train_max_samples
                )

            # pass a list of validation datasets
            self.valset = []
            i = 0
            for valset_name in self.mix_dataset_config:
                single_val_dataset_config = {valset_name: self.mix_dataset_config[valset_name]}
                valset_i = MixAudioDataset(
                    mix_dataset_config=single_val_dataset_config,  # instead of self.mix_dataset_config
                    split='validation',
                    segment_length=self.segment_length,
                    sampling_rate=self.sampling_rate,
                    transforms_aug=self.transforms_aug,
                    transforms_gt=self.transforms_gt,
                    eval_transforms_aug = self.eval_transforms_aug,
                    evaluation_mode=True,
                    max_samples = self.val_max_samples
                )
                if len(valset_i) > 0:
                    self.valset.append(valset_i)
                    print("valset_{}: {}".format(i, valset_name))
                    i += 1

        elif stage == "validation":
            self.valset = []
            for valset_name in self.mix_dataset_config:
                single_val_dataset_config = {valset_name: self.mix_dataset_config[valset_name]}
                valset_i = MixAudioDataset(
                    mix_dataset_config=single_val_dataset_config,  # instead of self.mix_dataset_config
                    split='validation',
                    segment_length=self.segment_length,
                    sampling_rate=self.sampling_rate,
                    transforms_aug=self.transforms_aug,
                    transforms_gt=self.transforms_gt,
                    eval_transforms_aug = self.eval_transforms_aug,
                    evaluation_mode=True,
                    max_samples = self.val_max_samples
                )
                if len(valset_i) > 0:
                    self.valset.append(valset_i)
                    print("valset_{}: {}".format(i, valset_name))
                    i += 1

        elif stage == "test":
            self.testset = []
            for testset_name in self.mix_dataset_config:
                single_test_dataset_config = {testset_name: self.mix_dataset_config[testset_name]}
                testset_i = MixAudioDataset(
                    mix_dataset_config=single_test_dataset_config,  # instead of self.mix_dataset_config
                    split='test',
                    segment_length=self.segment_length,
                    sampling_rate=self.sampling_rate,
                    transforms_aug=self.transforms_aug,
                    transforms_gt=self.transforms_gt,
                    eval_transforms_aug = self.eval_transforms_aug,
                    evaluation_mode=True,
                    max_samples = self.val_max_samples
                )
                if len(testset_i) > 0:
                    self.testset.append(testset_i)
                    print("testset_{}: {}".format(i, testset_name))
                    i += 1
        elif stage == "predict":
            self.predict_dataset = FullSequencePredictDataset(
                audio_file_list=self.predict_filelist,
                sampling_rate=self.sampling_rate,
                transforms_aug=self.transforms_aug,
                transforms_gt=self.transforms_gt,
                start_idx=self.predict_start_idx,
                end_idx=self.predict_end_idx
            )
        else:
            raise ValueError("Unimplemented stage in datamodule class")



    def train_dataloader(self):
        train_loader = DataLoader(
            self.trainset, num_workers=self.num_workers, shuffle=True,
            batch_size=self.batch_size, pin_memory=False, drop_last=True)
        return train_loader


    def val_dataloader(self):
        val_loader = []
        for valset_i in self.valset:
            val_loader.append(
                DataLoader(
                    valset_i, # self.valset
                    num_workers=self.num_workers, shuffle=False,
                    batch_size=self.batch_size, pin_memory=False, drop_last=False
                )
            )
        return val_loader


    def test_dataloader(self):
        print("initializing test dataloader")
        test_loader = []
        for testset_i in self.testset:
            test_loader.append(
                DataLoader(
                    testset_i,
                    num_workers=self.num_workers, shuffle=False,
                    batch_size=self.batch_size, pin_memory=False, drop_last=False
                )
            )
        return test_loader


    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=0, shuffle=False,
            batch_size=1, pin_memory=False,
            drop_last=False)
