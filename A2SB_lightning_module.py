# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Callback
from utils import SequenceLength, average_key_value
from typing import Optional, Dict, List
import numpy as np
import json
from diffusion import Diffusion, compute_gaussian_product_coef, get_multidiffusion_vf, multidiffusion_pad_inputs, multidiffusion_unpad_outputs
from networks import SinusoidalTemporalEmbedding
from plotting_utils import plot_spec_to_numpy
import torchaudio
import inspect
from audio_utils import phase_channels_to_R, stft_mag_R_to_wav, phase_R_to_channels
from audio_transforms.transforms import apply_audio_transforms
import ssr_eval
from collections import defaultdict, OrderedDict
import copy
import os
from scipy.io.wavfile import write as write_wav
from utils import find_middle_of_zero_segments

from tqdm import tqdm



class TimePartitionedPretrainedSTFTBridgeModel(LightningModule):
    def __init__(self, vf_model: torch.nn.Module,
                 inv_transforms=[], sampling_rate=22050,
                 n_timestep_channels=128,
                 beta_max=0.3, use_ot_ode=False,
                 fast_inpaint_mode=False,
                 pretrained_checkpoints: List[str]=None,
                 t_cutoffs: List[float]=[0.5],
                 predict_n_steps=50,
                 predict_hop_length=128,
                 predict_win_length=256,
                 predict_batch_size=16,
                 predict_output_dir="output"
                 ):
        super().__init__()
        for item in inspect.signature(TimePartitionedPretrainedSTFTBridgeModel).parameters:
            setattr(self, item, eval(item))
        self.ddpm = Diffusion(beta_max=beta_max)
        # SimpleUnet
        self.t_to_emb = SinusoidalTemporalEmbedding(n_bands=int(n_timestep_channels//2), min_freq=0.5)
        if self.use_ot_ode:
            print("Using ODE formulation")
        else:
            print("Using SDE formulation")
        self.test_results = defaultdict(list)
        assert(len(t_cutoffs) + 1 == len(pretrained_checkpoints))
        self.load_t_bounded_checkpoints(pretrained_checkpoints, t_cutoffs)
        self.fast_inpaint_mode = fast_inpaint_mode

    @torch.no_grad()
    def load_t_bounded_checkpoints(self, pretrained_checkpoints, t_cutoffs):
        loaded_models = []
        for ckpt in pretrained_checkpoints:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict'] # loads full PTL state dict
            # we just need the vf_model
            # rename keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'vf_model' in key: # skip other modules
                    new_key = key.replace('vf_model.', '')
                    new_state_dict[new_key] = value
            current_model = copy.deepcopy(self.vf_model)
            current_model.load_state_dict(new_state_dict)
            loaded_models.append(current_model)
        self.t_bounded_pretrained_models = nn.ModuleList(loaded_models)

    @torch.no_grad()
    def get_vf_model(self, t: float):
        model_idx = 0
        for idx, thresh in enumerate(self.t_cutoffs):
            if t >= thresh:
                model_idx = idx + 1
        return self.t_bounded_pretrained_models[model_idx]

    @torch.no_grad()
    def vocode_stft(self, spec_out):
        """
        # TODO move this outside of lightningmodule
        spec_out: B x C x H x W model outputs to be mapped back to waveform
        """
        all_wav_samples = [] # assume transforms don't support batch dimension for now
        num_samples = spec_out.shape[0]

        for b in range(num_samples):
            all_wav_samples.append(apply_audio_transforms(spec_out[b], self.inv_transforms)[0])

        return all_wav_samples

    @torch.no_grad()
    def ddpm_sample(self, x_1, t_steps=None, mask=None, mask_pred_x0=True,
                    win_length=256,
                    hop_length=256,
                    batch_size=16
                    ):
        """
        win_length: temporal window length of input spectrogram
        hop_length: step size. If hop_length < win_length, we use multidiffusion
        """
        assert hop_length <= win_length
        n_steps = t_steps.shape[1] - 1
        original_width = x_1.shape[-1]
        x_1 = multidiffusion_pad_inputs(x_1, win_length, hop_length)
        mask = multidiffusion_pad_inputs(mask, win_length, hop_length)

        x_t = x_1.clone()
        pred_x0 = None
        all_pred_x0s = []

        for t_idx in tqdm(range(n_steps)):
            # print(t_idx)
            t_emb = self.t_to_emb(t_steps[:,t_idx]).repeat(x_1.shape[0], 1)
            t = t_steps[:, t_idx]
            t_prev = t_steps[:, t_idx+1]
            #vf_output = self.get_vf_model(t[0].item())(x_t, t_emb)
            vf_model = self.get_vf_model(t[0].item())
            vf_output = get_multidiffusion_vf(vf_model, x_t, t_emb, win_length=win_length,
                                              hop_length=hop_length, batch_size=batch_size
                                              )
            pred_x0 = self.ddpm.get_pred_x0(t_steps[:, t_idx], x_t, vf_output)
            if mask is not None and mask_pred_x0:
                pred_x0 = pred_x0 * mask + (1-mask) * x_1

            all_pred_x0s.append(pred_x0.cpu())
            x_t_prev = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=self.use_ot_ode)
            x_t = x_t_prev
            if mask is not None:
                xt_true = x_1
                if not self.use_ot_ode:
                    std_sb = self.ddpm.get_std_t(t_prev)
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                x_t = (1. - mask) * xt_true + mask * x_t
        all_pred_x0s = [multidiffusion_unpad_outputs(pred, original_width) for pred in all_pred_x0s]
        return all_pred_x0s
    
    @torch.no_grad()
    def fast_inpaint_ddpm_sample(self, x_1, t_steps=None, mask=None, mask_pred_x0=True,
                    win_length=256,
                    hop_length=256,
                    batch_size=16):
        """
        assumes any masked segment is shorter win_length and sufficiently suparated
        """
        original_width = x_1.shape[-1]
        x_1 = x_1.clone()
        x_1 = multidiffusion_pad_inputs(x_1, win_length, hop_length)
        mask = multidiffusion_pad_inputs(mask, win_length, hop_length, padding_constant=0)

        middle_indices = find_middle_of_zero_segments(1-mask[0,0,0])
        for center_idx in middle_indices:
            l_idx = int(center_idx-win_length/2)
            r_idx = int(center_idx +win_length/2)
            if l_idx < 0:
                r_idx -= l_idx
                l_idx = 0

            if r_idx > x_1.shape[-1]:
                l_idx -= (r_idx - x_1.shape[-1])
                r_idx = x_1.shape[-1]
            assert(r_idx - l_idx == win_length)
            assert(l_idx >= 0)
            assert(r_idx <= x_1.shape[-1])
            curr_x_1 = x_1[:,:,:,l_idx:r_idx]
            curr_mask = mask[:,:,:,l_idx:r_idx]
            new_x_0 = self.ddpm_sample(curr_x_1, t_steps, mask=curr_mask, mask_pred_x0=mask_pred_x0, win_length=win_length, hop_length=hop_length, batch_size=batch_size)
            x_1[:,:,:,l_idx:r_idx] = new_x_0[-1]
        x_1 = multidiffusion_unpad_outputs(x_1, original_width)
        return [x_1]
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        bs, channels, height, width = batch['x_0_clean'].shape
        assert(bs == 1) # only supports batch size 1 for now

        current_out_dir = os.path.join(self.predict_output_dir, batch['outdir'][0])
        os.makedirs(current_out_dir, exist_ok=True)
        x_0_clean = batch['x_0_clean']
        x_0_corrupted = batch['x_0_corrupted']
        mask = batch['loss_mask'] # fill-mask
        t_steps = torch.linspace(1,0.05, int(self.predict_n_steps)).unsqueeze(0).to(x_0_corrupted.device)
        if not self.fast_inpaint_mode:
            x_0s = self.ddpm_sample(x_0_corrupted, t_steps=t_steps, mask=mask,
                                    mask_pred_x0=True, win_length=self.predict_win_length, hop_length=self.predict_hop_length,
                                    batch_size=self.predict_batch_size)
        else:
            x_0s = self.fast_inpaint_ddpm_sample(x_0_corrupted, t_steps=t_steps, mask=mask,
                                    mask_pred_x0=True, win_length=self.predict_win_length, hop_length=self.predict_hop_length,
                                    batch_size=self.predict_batch_size)

        reconstructed_audio = self.vocode_stft(x_0s[-1].cpu())[0].cpu().data.numpy()
        input_audio = self.vocode_stft(x_0_corrupted.cpu())[0].cpu().data.numpy()
        write_wav(os.path.join(current_out_dir, "recon.wav"), batch['output_sr'], reconstructed_audio)
        write_wav(os.path.join(current_out_dir, "dirty.wav"), batch['output_sr'], input_audio)


class STFTBridgeModel(LightningModule):
    def __init__(self, 
        vf_model: torch.nn.Module, 
        learning_rate=1e-4, weight_decay=1e-8, 
        inv_transforms=[], sampling_rate=22050,
        n_timestep_channels=128,
        beta_max=0.3, use_ot_ode=False,
        train_t_min=0, train_t_max=1
    ):
        super().__init__()
        for item in inspect.signature(STFTBridgeModel).parameters:
            setattr(self, item, eval(item))
        self.ddpm = Diffusion(beta_max=beta_max)

        # SimpleUnet
        self.t_to_emb = SinusoidalTemporalEmbedding(n_bands=int(n_timestep_channels//2), min_freq=0.5)
        assert train_t_max > train_t_min
        
        # Transformer UNet
        # self.t_to_emb = lambda x: x

        self.inv_transforms = inv_transforms # inverse transforms to map spectrograms back to waveform
        self.sampling_rate = sampling_rate # needed for vocoding
        self.use_ot_ode = use_ot_ode
        if self.use_ot_ode:
            print("Using ODE formulation")
        else:
            print("Using SDE formulation")
        self.test_results = defaultdict(list)


    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay, decoupled_weight_decay=True)
        return optimizer

    def ddpm_sample(self, x_1, t_steps=None, mask=None, mask_pred_x0=True):
        n_steps = t_steps.shape[1] - 1

        x_t = x_1.clone()
        pred_x0 = None
        all_pred_x0s = []

        for t_idx in range(n_steps):

            t_emb = self.t_to_emb(t_steps[:,t_idx]).repeat(x_1.shape[0], 1)
            t = t_steps[:, t_idx] 
            t_prev = t_steps[:, t_idx+1] 
            vf_output = self.vf_model(x_t, t_emb)
            pred_x0 = self.ddpm.get_pred_x0(t_steps[:, t_idx], x_t, vf_output)
            if mask is not None and mask_pred_x0:
                pred_x0 = pred_x0 * mask + (1-mask) * x_1

            all_pred_x0s.append(pred_x0.cpu())
            x_t_prev = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=self.use_ot_ode)

            x_t = x_t_prev
            if mask is not None:
                xt_true = x_1
                if not self.use_ot_ode:
                    # if t_idx == 0:
                    #     print('sampling using ddpm')
                    std_sb = self.ddpm.get_std_t(t_prev)
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # else:
                #     if t_idx == 0:
                #         print('sampling using otode')

                x_t = (1. - mask) * xt_true + mask * x_t

        return all_pred_x0s
    
    def ddpm_sample_i2sb_way(self, x_1, t_steps=None, mask=None):
        n_steps = t_steps.shape[1] - 1

        x_t = x_1.clone()
        pred_x0 = None
        all_pred_x0s = []

        for t_idx in range(n_steps):

            t_emb = self.t_to_emb(t_steps[:,t_idx]).repeat(x_1.shape[0], 1)
            t = t_steps[:, t_idx]
            t_prev = t_steps[:, t_idx+1]

            vf_output = self.vf_model(x_t, t_emb)
            pred_x0 = self.ddpm.get_pred_x0(t_steps[:, t_idx], x_t, vf_output)
            x_t = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=self.use_ot_ode)

            if mask is not None:
                xt_true = x_1
                if not self.use_ot_ode:
                    # if t_idx == 0:
                    #     print('sampling using ddpm')

                    sigma_fwd = self.ddpm.get_std_fwd(t_prev)
                    sigma_rev = self.ddpm.get_std_rev(t_prev)
                    coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
                    std_sb = torch.sqrt(var)
                    
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # else:
                #     if t_idx == 0:
                #         print('sampling using otode')
                
                x_t = (1. - mask) * xt_true + mask * x_t

            all_pred_x0s.append(pred_x0.cpu())
        return all_pred_x0s

    def ddpm_sample_i2sb_change_order(self, x_1, t_steps=None, mask=None):
        n_steps = t_steps.shape[1] - 1

        x_t = x_1.clone()
        pred_x0 = None
        all_pred_x0s = []

        for t_idx in range(n_steps):

            t_emb = self.t_to_emb(t_steps[:,t_idx]).repeat(x_1.shape[0], 1)
            t = t_steps[:, t_idx] 
            t_prev = t_steps[:, t_idx+1] 
            vf_output = self.vf_model(x_t, t_emb)
            pred_x0 = self.ddpm.get_pred_x0(t_steps[:, t_idx], x_t, vf_output)

            if mask is not None:
                xt_true = x_1
                if not self.use_ot_ode:
                    # if t_idx == 0:
                    #     print('sampling using ddpm')
                        
                    sigma_fwd = self.ddpm.get_std_fwd(t_prev)
                    sigma_rev = self.ddpm.get_std_rev(t_prev)
                    coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
                    std_sb = torch.sqrt(var)
                    
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # else:
                #     if t_idx == 0:
                #         print('sampling using otode')

                pred_x0 = pred_x0 * mask + (1-mask) * xt_true

            all_pred_x0s.append(pred_x0.cpu())
            x_t_prev = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=self.use_ot_ode)

            x_t = x_t_prev
        return all_pred_x0s
    
    def vocode_stft(self, spec_out):
        """
        # TODO move this outside of lightningmodule
        spec_out: B x C x H x W model outputs to be mapped back to waveform
        """
        all_wav_samples = [] # assume transforms don't support batch dimension for now
        num_samples = spec_out.shape[0]

        for b in range(num_samples):
            all_wav_samples.append(apply_audio_transforms(spec_out[b], self.inv_transforms)[0])

        return all_wav_samples
    
    def sample_t_bounded(self, n_samples):
        t_range = self.train_t_max - self.train_t_min
        return torch.rand(n_samples) * t_range + self.train_t_min

    def training_step(self, batch, batch_idx):
        self.log('global_step', int(self.global_step))
        x_0_clean = batch['x_0_clean']
        x_0_corrupted = batch['x_0_corrupted']
        loss_mask = batch['loss_mask']
        t = self.sample_t_bounded(x_0_clean.shape[0]).to(x_0_clean.device)
        t_emb = self.t_to_emb(t)
        x_t = self.ddpm.q_sample(t, x_0_clean, x_0_corrupted, ot_ode=self.use_ot_ode)
        std_fwd_t = self.ddpm.get_std_fwd(t)
        vf_output = self.vf_model(x_t, t_emb)
        # apply mask
        std_fwd_t_expanded = std_fwd_t.view(std_fwd_t.shape[0], 1, 1, 1)
        target = (x_t - x_0_clean)/std_fwd_t_expanded
        loss_mask = batch['loss_mask'] if 'loss_mask' in batch.keys() else 1
        loss = (((vf_output - target.detach()) ** 2) * loss_mask).sum() / loss_mask.sum()

        # store loss
        self.log("train/loss", loss)
        if torch.isnan(loss):
            print("Loss is NaN. Skipping this batch.")
            return None

        print('\nloss: {:.4f}'.format(loss.cpu().item()))

        # store gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.log("train/gradient_norm", grad_norm)
        
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x_0_clean = batch['x_0_clean']
        x_0_corrupted = batch['x_0_corrupted']
        loss_mask = batch['loss_mask']
        n_steps = 25
        n_samples = x_0_clean.shape[0]
        t_steps = torch.linspace(1, 1.0/n_steps, int(n_steps)).unsqueeze(0).to(x_0_corrupted.device)

        x_0s = self.ddpm_sample(x_0_corrupted, t_steps=t_steps, mask=loss_mask, mask_pred_x0=True)
        inpainted_audio = self.vocode_stft(x_0s[-1].cpu())
        metrics = ssr_eval.metrics.AudioMetrics(rate=self.sampling_rate)

        for idx in range(n_samples):
            result = metrics.evaluation(inpainted_audio[idx].cpu().numpy(), batch['x_0_wav'][idx].cpu().numpy(), None)
            self.test_results[dataloader_idx].append(result)

    def on_test_end(self):
        for dataloader_idx, test_results_idx in self.test_results.items():
            print('---------- dataloader {} ----------'.format(dataloader_idx))
            for key in test_results_idx[0].keys():
                val = average_key_value(test_results_idx, key)
                print('average for {} is {:.3f}'.format(key, val))
        return self.test_results

    def predict_step(self, batch, batch_idx):
        breakpoint()
        print("hello")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.vf_model.eval()
        output_dict = {}

        try:
            x_0_clean = batch['x_0_clean']
            x_0_corrupted = batch['x_0_corrupted']
            loss_mask = batch['loss_mask'] if 'loss_mask' in batch.keys() else 1
            scales = [0.01, 0.25, 0.50, 0.75, 0.99]
            for scale in scales:
                t = scale * torch.ones(x_0_clean.shape[0]).to(x_0_clean.device)
                t_emb = self.t_to_emb(t)
                x_t = self.ddpm.q_sample(t, x_0_clean, x_0_corrupted, ot_ode=self.use_ot_ode)
                std_fwd_t = self.ddpm.get_std_fwd(t)
                vf_output = self.vf_model(x_t, t_emb)
                # apply mask
                std_fwd_t_expanded = std_fwd_t.view(std_fwd_t.shape[0], 1, 1, 1)
                target = (x_t - x_0_clean)/std_fwd_t_expanded
                loss = (((vf_output - target.detach()) ** 2 ) * loss_mask).sum() / loss_mask.sum()
                self.log("val_loss_t={}".format(scale), loss)
            
            n_steps = 25
            n_samples = x_0_clean.shape[0]
            t_steps = torch.linspace(1, 1.0/n_steps, int(n_steps)).unsqueeze(0).to(x_0_corrupted.device)

            x_0s = self.ddpm_sample(x_0_corrupted, t_steps=t_steps, mask=loss_mask, mask_pred_x0=True)
            inpainted_audio = self.vocode_stft(x_0s[-1].cpu())
            metrics = ssr_eval.metrics.AudioMetrics(rate=self.sampling_rate)
            results = []
            for idx in range(n_samples):
                result = metrics.evaluation(inpainted_audio[idx].cpu().numpy(), batch['x_0_wav'][idx].cpu().numpy(), None)
                results.append(result)
            
            for key in results[0].keys():
                val = average_key_value(results, key)
                output_dict[key] = val
                self.log(f"val_{key}", val)

        except Exception as e:
            print("Error:", e)

        self.vf_model.train()
        return output_dict


class LogValidationInpaintingSTFTCallback(Callback):
    def get_mag(self, spec):
        # print('DEBUG: spec.shape', spec.shape)  # [1, 3, 1024, 256]
        if spec.shape[-3] == 2:
            return torch.sqrt((spec ** 2).sum(-3))
        else:
            return spec[:,0]
            # return spec[...,0,:,:]


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        
        val_dataset = trainer.val_dataloaders[0].dataset
        num_to_plot = 0
        if pl_module.global_rank == 0 and batch_idx == 0:
            for i in range(num_to_plot):
                sample = val_dataset[i]
                sample_id = "Validation sample " + str(i) + "/"
                seq_lens = sample['seq_lens']

                x_0_corrupted = sample['x_0_corrupted'].unsqueeze(0).to(pl_module.device)
                x_0_corrupted_mag = pl_module.inv_transforms[0](self.get_mag(x_0_corrupted))
                x_0_clean = sample['x_0_clean'].unsqueeze(0).to(pl_module.device) # otherwise some asynch transforms don't get run
                x_0_clean_mag = pl_module.inv_transforms[0](self.get_mag(x_0_clean))
                mask = sample['loss_mask'].unsqueeze(0).to(pl_module.device)
                # print('DEBUG: x_0_clean.shape', x_0_clean.shape)  # [1, 3, 1024, 256]
                gt_reconstruction = pl_module.vocode_stft(x_0_clean.unsqueeze(0).cpu())

                pl_module.logger.experiment.add_audio(sample_id + "Original Audio",
                                                      gt_reconstruction[0].data.numpy(), pl_module.global_step,
                                                      pl_module.sampling_rate)

                pl_module.logger.experiment.add_image(sample_id + "Original Magnitude",
                                                      plot_spec_to_numpy(x_0_clean_mag[0].data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")
                pl_module.logger.experiment.add_image(sample_id + "Masked Magnitude",
                                                      plot_spec_to_numpy(x_0_corrupted_mag[0].data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")

                n_steps = 25
                t_steps = torch.linspace(1,0.05, n_steps).unsqueeze(0).to(x_0_corrupted.device)
                x_0s = pl_module.ddpm_sample(x_0_corrupted.unsqueeze(0), mask=mask, t_steps=t_steps)
                sampled_spec = x_0s[-1]
                sampled_spec_mag = self.get_mag(pl_module.inv_transforms[0](sampled_spec[0]))
                pl_module.logger.experiment.add_image(sample_id + "Inpainted Magnitude",
                                                      plot_spec_to_numpy(sampled_spec_mag.data.cpu().numpy()),
                                                      pl_module.global_step, dataformats="HWC")
                inpainted_audio = pl_module.vocode_stft(sampled_spec[0:1].cpu())
                pl_module.logger.experiment.add_audio(sample_id + "Inpainted Audio",
                                                      inpainted_audio[0].data.numpy(), pl_module.global_step,
                                                      pl_module.sampling_rate)
