# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
from torch import Tensor

from typing import List, Optional, Tuple, Union


def radian_to_SO2(rads: torch.Tensor):
    """
    converts tensor of radians to tensor of SO(2) matrices
    for any inputs tensor of shape B x ..., the output will be B x ... x 2 x 2
    """
    cos_theta = torch.cos(rads)
    sin_theta = torch.sin(rads)

    rot_m = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], -1)
    rot_m = rot_m.view(*rot_m.shape[:-1], 2, 2)
    return rot_m


def wav_to_stft(wav: torch.Tensor, fft_size: int, hop_size: int, win_length: int, drop_dc_term=True):
    """
    Inputs:
        wav: N or B x N tensor of waveform sample values
    Returns:
        magnitude and phase tensors with or without a batch dimension depending on wav input
        Magnitudes: ... x H x W
        phase_R: ... x H x W x 2 x 2 (represented as SO2 matrices)
    """

    stft_cmplx = torch.stft(wav, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=torch.hann_window(win_length), return_complex=True)
    if drop_dc_term:
        stft_cmplx = stft_cmplx[...,1:,:]

    magnitudes = stft_cmplx.abs() # abs on complex values should compute the vector norm
    stft_real = torch.view_as_real(stft_cmplx)
    phase = torch.atan2(stft_real[...,1],stft_real[...,0])
    phase_R = radian_to_SO2(phase)
    return magnitudes, phase_R

def phase_R_to_channels(stft_R):
    """
    converts B x H x W x 2 x 2 to B x 4 x H x W
    """
    if len(stft_R.shape) == 5:
        return stft_R.reshape(*stft_R.shape[:3], 4).permute(0,3,1,2)
    elif len(stft_R.shape) == 4:
        return phase_R_to_channels(phase_R_to_channels.unsqueeze(-1))[0]
    else:
        print("unsupported dimensions")
        exit(1)


def phase_channels_to_R(stft_channels):
    """
    inverse transformation for phase_R_to_channels
    """
    stft_R_flat = stft_channels.permute(0, 2, 3, 1)
    stft_R = stft_R_flat.reshape(*stft_R_flat.shape[:3], 2, 2)
    return stft_R


def stft_mag_R_to_wav(stft_mag, stft_Rch, n_fft, hop_length, win_length, append_dc_term=True):
    """
    stft_mag: B x 1 x H x L magnitudes
    stft_R: B x 4 x H x L flattened rotations
    """
    stft_costheta = stft_Rch[:,0:1]
    stft_sintheta = stft_Rch[:,2:3]
    x_cos_theta = stft_costheta * stft_mag
    x_sin_theta = stft_sintheta * stft_mag

    new_stft_cmplx = torch.view_as_complex(torch.stack([x_cos_theta[:,0], x_sin_theta[:,0]], -1))
    if append_dc_term: # if dc term was removed, add it back in with zero magnitude, zero phase
        _b, _h, _w = new_stft_cmplx.shape
        dct = torch.zeros(_b, 1, _w, device=new_stft_cmplx.device)
        new_stft_cmplx = torch.cat((dct, new_stft_cmplx), 1)
    wav_out = torch.istft(new_stft_cmplx, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length).to(new_stft_cmplx.device))
    return wav_out


def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")
