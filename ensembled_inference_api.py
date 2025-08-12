# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from lightning.pytorch.cli import LightningCLI
from A2SB_lightning_module_api import TimePartitionedPretrainedSTFTBridgeModel, LogValidationInpaintingSTFTCallback
from datasets.datamodule import STFTAudioDataModule
from lightning.pytorch.callbacks import ModelCheckpoint


class InpaintingAudioSBLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.add_lightning_class_args(LogValidationInpaintingSTFTCallback, "validation_inpainting_callback")
        parser.set_defaults({"checkpoint_callback.filename": "latest-epoch_{epoch}-iter_{global_step:.0f}",
                             "checkpoint_callback.monitor": "global_step",
                             "checkpoint_callback.mode": "max",
                             "checkpoint_callback.every_n_train_steps": 1000,
                             "checkpoint_callback.dirpath": "/debug",
                             "checkpoint_callback.save_top_k": -1,
                             "checkpoint_callback.auto_insert_metric_name": False})
        parser.link_arguments("checkpoint_callback.dirpath", "trainer.default_root_dir")

        # parser.link_arguments("data.fft_size", "model.fft_size")
        # parser.link_arguments("data.hop_size", "model.hop_size")
        # parser.link_arguments("data.win_length", "model.win_length")
        # parser.link_arguments("data.sampling_rate", "model.sampling_rate")
if __name__ == '__main__':
        cli = InpaintingAudioSBLightningCLI(TimePartitionedPretrainedSTFTBridgeModel, STFTAudioDataModule, save_config_kwargs={"overwrite": True})
