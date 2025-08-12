# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


# Original source and license:
# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB.
# https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py


import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from math import ceil
from einops import rearrange


def get_multidiffusion_vf(vf_model, x_t, t_emb, win_length=256, hop_length=128, batch_size=16):
    """
    t_emb should be b x emb_dim but all embeddiengs should be for the same time step, as this code does not
    support heterogenous sampling schedulings
    """
    b_size, num_channels, win_height, seq_len = x_t.shape
    counts = torch.zeros_like(x_t)
    num_hops = (seq_len - (win_length - hop_length))//hop_length
    l_idx = 0
    vf_t = torch.zeros_like(x_t)
    unfolder = torch.nn.Unfold([win_height, win_length], stride=hop_length)
    x_t_unfolded = unfolder(x_t)
    num_channels = x_t.shape[1]
    b_size = x_t.shape[0]
    x_t_unfolded = rearrange(x_t_unfolded, "b (c h w) l -> (b l) c h w", c=num_channels,
                             h=win_height, w=win_length)
    # compute the vector fields in batches
    num_chunks = ceil(x_t_unfolded.shape[0]/batch_size)
    x_t_unfolded_chunked = torch.chunk(x_t_unfolded, num_chunks)
    vfields_out = []
    t_emb_rpt = t_emb.repeat(num_hops, 1)
    t_emb_chunked = torch.chunk(t_emb_rpt, num_chunks)
    for b_chunk_idx in range(num_chunks):
        vfields_out.append(vf_model(x_t_unfolded_chunked[b_chunk_idx], t_emb_chunked[b_chunk_idx]))
    vfields = torch.cat(vfields_out, 0)
    vfields = rearrange(vfields, "(b l) c h w -> l b c h w", b=b_size, l=num_hops)

    for hop_idx in range(int(num_hops)):

        r_idx = l_idx + win_length
        counts[...,l_idx:r_idx]+=1
        curr_x_t = x_t[...,l_idx:r_idx]
        vf_out = vfields[hop_idx] #vf_model(curr_x_t, t_emb)

        vf_t[...,l_idx:r_idx] += vf_out
        l_idx += hop_length

    return vf_t / counts


def multidiffusion_pad_inputs(input, win_length, hop_length, padding_constant=None):
    _b, _c, _h, width = input.shape
    if width <= win_length: # no hops
        to_pad = win_length - width
    else:
        pad_to = ceil((width - win_length)/ hop_length) * hop_length + win_length
        to_pad = pad_to - width

    if to_pad > 0:
        padding = input[..., :to_pad]
        if padding_constant is not None:
            padding = padding*0+padding_constant

        input_padded = torch.cat([input, padding], dim=-1)
    else:
        input_padded = input.clone()
    return input_padded


def multidiffusion_unpad_outputs(output, original_width: int):
    return output[...,:original_width]


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion(nn.Module):
    def __init__(self, beta_min=1e-4, beta_max=0.3):
        super().__init__()
        # t = 0 (clean data), t=1 (corrputed posterior)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta_t(self, t):
        # beta = sqrt(t)*beta/0.5
        if t <= 0.5:
            return t**2 * self.beta_max
        else:
            return (1-t)**2 * self.beta_max

    def get_int_beta_0_t(self, t):
        """
        t: torch.tensor [0,1]
        """
        beta_int = t.clone()
        full_integral = 2 * self.beta_max*(0.5**3)/3
        half_inds = t > 0.5
        beta_int[half_inds] = full_integral - 1/3*self.beta_max * ((1-t[half_inds])**3)
        beta_int[~half_inds] = 1/3*self.beta_max * (t[~half_inds]**3)
        return beta_int

    def get_std_fwd(self, t):
        return torch.sqrt(self.get_int_beta_0_t(t))

    def get_std_rev(self, t):
        return torch.sqrt(self.get_int_beta_0_t(1-t))

    def get_std_t(self, t):
        sigma_fwd = self.get_std_fwd(t)
        sigma_rev = self.get_std_rev(t)
        coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
        return torch.sqrt(var)

    def q_sample(self, t, x_0, x_1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """
        sigma_fwd = self.get_std_fwd(t)
        sigma_rev = self.get_std_rev(t)

        coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
        while len(coef1.shape) < len(x_0.shape):
            coef1 = coef1[:, None]
            coef2 = coef2[:, None]
            var = var[:, None]
        x_t = coef1 * x_0 + coef2 * x_1
        std_sb_t = torch.sqrt(var)
        if not ot_ode:
            x_t += std_sb_t * torch.randn_like(x_t)
        return x_t.detach()

    def p_posterior(self, t_prev, t, x_t, x_0, ot_ode=False):
        assert t_prev < t
        std_t = self.get_std_fwd(t)
        std_t_prev =  self.get_std_fwd(t_prev)
        std_delta = (std_t**2 - std_t_prev**2).sqrt()
        mu_x0, mu_xt, var = compute_gaussian_product_coef(std_t_prev, std_delta)
        x_t_prev = mu_x0 * x_0 + mu_xt * x_t

        if not ot_ode and t_prev > 0:
            x_t_prev = x_t_prev + var.sqrt() * torch.randn_like(x_t_prev)
        return x_t_prev

    def get_pred_x0(self, t, x_t, net_out):
        std_fwd_t = self.get_std_fwd(t)
        pred_x0 = x_t - std_fwd_t * net_out
        return pred_x0
