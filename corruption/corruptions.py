# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
import numpy as np



def mask_with_noise(x, mask, noise_level):
    return x * (1-mask) + mask * torch.randn_like(x) * noise_level


class UpsampleMask:
    def __init__(self, min_cutoff_freq: int, max_cutoff_freq: int, sampling_rate: int, dc_dropped: bool=True):
        super().__init__()
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.sampling_rate = sampling_rate
        self.dc_dropped = dc_dropped

    @staticmethod
    def get_upsample_mask(spec: torch.Tensor, min_cutoff_freq: int, max_cutoff_freq: int, sampling_rate: int, dc_dropped=True):
        """
        input:
            spec: C x H x L spectrograms batched
        returns:
            mel_masked: C x H x L mel spectrograms batched, with areas filled in with white noise
        """

        c, h, l = spec.shape
        inpaint_mask = torch.zeros(c, h, l).to(spec.device)
        # low  = int(h * min_cutoff_freq / float(sampling_rate))
        # high = min(int(h * max_cutoff_freq / float(sampling_rate)), h)
        if dc_dropped:
            n_fft = h * 2
        else:
            n_fft = (h - 1) * 2
        inpaint_mask = torch.zeros(c, h, l).to(spec.device)
        low  = int(n_fft * min_cutoff_freq / float(sampling_rate))
        high = min(int(n_fft * max_cutoff_freq / float(sampling_rate)), h)
        high = max(high, low + 1)  # make sure high > low

        cutoff = torch.randint(low=low, high=high, size=[1])

        inpaint_mask[:, cutoff[0]:, :] = 1
        return inpaint_mask

    def __call__(self, spec: torch.Tensor):
        return self.get_upsample_mask(spec, self.min_cutoff_freq, self.max_cutoff_freq, self.sampling_rate, self.dc_dropped)


class ExtensionMask:
    def __init__(self, min_edge_distance=32):
        super().__init__()
        self.min_edge_distance = min_edge_distance

    @staticmethod
    def get_extension_mask(spec: torch.Tensor, min_edge_distance: int):
        """
        input:
            spec: C x H x L spectrograms batched
        returns:
            mel_masked: C x H x L mel spectrograms batched, with areas filled in with white noise
        """

        c, h, l = spec.shape
        inpaint_mask = torch.zeros(c, h, l).to(spec.device)
        mask_start_ind = torch.randint(low=min_edge_distance, high=l-min_edge_distance, size=[1])

        if torch.randn(1) > 0: # to the right
            inpaint_mask[:, :, mask_start_ind[0]:] = 1
        else: # to the left
            inpaint_mask[:, :, :mask_start_ind[0]] = 1
        return inpaint_mask

    def __call__(self, spec: torch.Tensor):
        return self.get_extension_mask(spec, self.min_edge_distance)


class InpaintMask:
    def __init__(self, min_inpainting_frac: float, max_inpainting_frac: float, is_random: bool):
        super().__init__()
        assert 0.0 <= min_inpainting_frac <= max_inpainting_frac <= 1.0
        self.min_inpainting_frac = min_inpainting_frac
        self.max_inpainting_frac = max_inpainting_frac
        self.is_random = is_random

    @staticmethod
    def get_inpainting_mask(spec: torch.Tensor, min_inpainting_frac, max_inpainting_frac, is_random):

        c, h, w = spec.shape
        # print('spec.shape', spec.shape)  # torch.Size([3, 1024, 256])
        inpaint_mask = torch.zeros(c, h, w).to(spec.device)

        random_variable_for_length = np.random.rand()
        inpainting_frac = random_variable_for_length * (max_inpainting_frac - min_inpainting_frac) + min_inpainting_frac
        if inpainting_frac == 0:
            # that is, min_inpainting_frac = max_inpainting_frac = 0.0
            return inpaint_mask
        
        if not is_random:
            inpainting_start_frac = 0.5 - inpainting_frac / 2.0
        else:
            inpainting_start_frac = np.random.rand() * (1.0 - inpainting_frac)
        
        inpainting_start = int(inpainting_start_frac * w)
        inpainting_end = int((inpainting_start_frac + inpainting_frac) * w)
        inpaint_mask[:, :, inpainting_start:inpainting_end] = 1
        return inpaint_mask

    def __call__(self, spec: torch.Tensor):
        return self.get_inpainting_mask(spec, self.min_inpainting_frac, self.max_inpainting_frac, self.is_random)


class MultinomialInpaintMaskTransform:
    def __init__(self, p_upsample_mask=0.5, p_extension_mask=0.5, p_inpaint_mask=0.0, fill_noise_level=0.5, sampling_rate=22050, upsample_mask_kwargs={}, inpainting_mask_kwargs={}):
        """
        TODO: include other parameters for individual transforms
        """
        self.mask_fns = [UpsampleMask(sampling_rate=sampling_rate, **upsample_mask_kwargs), ExtensionMask(), InpaintMask(**inpainting_mask_kwargs)]
        self.mask_multinomial_probs = torch.Tensor([p_upsample_mask, p_extension_mask, p_inpaint_mask])
        self.fill_noise_level = fill_noise_level


    def __call__(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            spec: spectrogram tensor of shape C x H x W
        Returns:
            masked_input: masked version of spec with noise-filled holes
            mask: C x H x W binary masked used
        """

        mask_fn = self.mask_fns[torch.multinomial(self.mask_multinomial_probs, 1)]
        mask = mask_fn(spec)

        masked_and_noised_spec = mask_with_noise(spec, mask, self.fill_noise_level)

        return masked_and_noised_spec, mask


class TimestampedSegmentInpaintMaskTransform:
    def __init__(self, start_time=0.5, end_time=1.0, hop_length=512, sampling_rate=44100, fill_noise_level=0.5):
        self.start_idx = int(sampling_rate/hop_length*start_time)
        self.end_idx = int(sampling_rate/hop_length*end_time)
        self.fill_noise_level = fill_noise_level

    def __call__(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        spec: C x H x W
        """
        mask = torch.zeros_like(spec)
        mask[:, :, self.start_idx:self.end_idx] = 1
        masked_and_noised_spec = mask_with_noise(spec, mask, self.fill_noise_level)
        return masked_and_noised_spec, mask
