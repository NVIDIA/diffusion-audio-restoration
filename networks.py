# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


# Implemented based on Guided Diffusion https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
#     Licensed under the MIT license.


import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union, List
from utils import SequenceLength
from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
    )
from abc import abstractmethod


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class EmbeddingConditionalBlock(nn.Module):
    """
    Any module where forward() takes an arbitrary embedding as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class EmbeddingConditionalSequential(nn.Sequential, EmbeddingConditionalBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbeddingConditionalBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(EmbeddingConditionalBlock):
    def __init__(self, in_channels: int, out_channels: int, emb_in_channels: int, use_scale_shift_norm=True, use_skip=True, p_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_skip = use_skip
        if self.use_skip:
            assert(in_channels == out_channels)

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(emb_in_channels, self.out_channels * 2 if use_scale_shift_norm else self.out_channels, kernel_size=1)
            )
        self.out_norm = GroupNorm32(32, out_channels)
        self.out_rest = nn.Sequential(nn.SiLU(),
                                      nn.Dropout(p=p_dropout),
                                      zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))


    def forward(self, x, emb):
        """
        x: torch.tensor B x in_channels x H x W
        emb: B x emb_in_channels x H x W
        """

        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)#.unsqueeze(-1).unsqueeze(-1)
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = h + emb_out
            h = self.out_rest(self.out_norm(h))

        if self.use_skip:
            return x + h
        else:
            return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttnUNetF(nn.Module):
    def __init__(self, n_updown_levels: int, in_channels: int, hidden_channels: Union[int, List[int]], out_channels: int, emb_channels: int,
                 rotary_dims=16, band_embedding_dim=0, attention_levels=None, n_attn_heads=4, num_res_blocks=2, use_attn_input_norm=True):
        """
        Final architecture with sane parameterization
        inputs:
            attention_levels: 0-indexed levels specifying which ones should have attention layers
            n_updown_levels: total number of levels
            use_attn_input_norm: whether to use gradnorm as input to attention layers. Defaults to false, but will default to True in the future
            num_res_blocks: number of computational blocks per level
        """
        super().__init__()
        self.band_embedding_dim = band_embedding_dim
        assert(band_embedding_dim%2 == 0)
        self.enc_blocks = nn.ModuleList()
        self.ds_layers = nn.ModuleList()
        self.us_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attention_levels = torch.tensor(attention_levels)
        decoder_attention_levels = n_updown_levels - 1 - self.attention_levels # index for decoder half

        self.n_updown_levels = n_updown_levels
        if type(hidden_channels) is int:
            self.hidden_channels_levels = [hidden_channels] * (n_updown_levels + 1)

        else:
            self.hidden_channels_levels = hidden_channels
        hidden_channels_levels = self.hidden_channels_levels
        self.input_projection = nn.Conv2d(in_channels, hidden_channels_levels[0], 3, padding=1)
        self.emb_channels = emb_channels + band_embedding_dim

        for level in range(n_updown_levels):
            # construct encoder
            # level 0 is the input layer
            ds_in_channels = hidden_channels_levels[level]
            ds_out_channels = hidden_channels_levels[level+1]
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ds_in_channels, ds_in_channels, self.emb_channels))
                if level in attention_levels:  # downsampling level
                    layers.append(RotaryAttentionPool2d(embed_dim=ds_in_channels,
                                                        rotary_dim=32,
                                                        attn_dim=ds_in_channels,
                                                        num_heads=n_attn_heads,
                                                        output_dim=ds_in_channels,
                                                        use_input_norm=use_attn_input_norm
                                                        ))
            self.enc_blocks.append(EmbeddingConditionalSequential(*layers))
            self.ds_layers.append(Downsample(ds_in_channels, True, out_channels=ds_out_channels))

            # construct decoder, level 0 is the first level after the middle block here
            us_in_channels = hidden_channels_levels[n_updown_levels - level]
            us_out_channels = hidden_channels_levels[n_updown_levels - level - 1]
            self.us_layers.append(Upsample(us_in_channels, True, out_channels=us_out_channels))
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResBlock(us_in_channels, us_in_channels, self.emb_channels))
                if level in decoder_attention_levels:
                    layers.append(RotaryAttentionPool2d(embed_dim=us_in_channels,
                                                        rotary_dim=32,
                                                        attn_dim=us_in_channels,
                                                        num_heads=n_attn_heads,
                                                        output_dim=us_in_channels,
                                                        use_input_norm=use_attn_input_norm
                                                        ))
            self.dec_blocks.append(EmbeddingConditionalSequential(*layers))
        # construct middle block
        self.middle_block = EmbeddingConditionalSequential(ResBlock(hidden_channels_levels[-1],
                                                                     hidden_channels_levels[-1],
                                                                     self.emb_channels),
                                                           RotaryAttentionPool2d(embed_dim=hidden_channels_levels[-1],
                                                                                 rotary_dim=32,
                                                                                 attn_dim=hidden_channels_levels[-1],
                                                                                 num_heads=n_attn_heads,
                                                                                 output_dim=hidden_channels_levels[-1],
                                                                                 use_input_norm=use_attn_input_norm
                                                                                 ),
                                                           ResBlock(hidden_channels_levels[-1],
                                                                    hidden_channels_levels[-1],
                                                                    self.emb_channels))

        self.output_projection = nn.Sequential(
            GroupNorm32(32, hidden_channels_levels[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels_levels[0], out_channels, 3, padding=1)
            )

    def get_band_embeddings(self, n_bands, device):
        n_freqs = self.band_embedding_dim // 2
        coords = torch.arange(0, n_bands).to(device)
        coords_exp = coords.unsqueeze(0).repeat(self.band_embedding_dim//2, 1)
        freqs = torch.arange(0, n_freqs).unsqueeze(-1) + 1
        freqs = freqs.to(device)
        coords_exp = freqs * (coords_exp) *2*3.14/ (3*n_bands) # this way longet period should exceed n_bands
        cos_embs = torch.cos(coords_exp)
        sin_embs = torch.sin(coords_exp)
        band_embs = torch.cat((cos_embs, sin_embs), 0)
        band_embs_cat = band_embs.unsqueeze(0).unsqueeze(-1)
        return band_embs_cat

    def forward(self, x, emb):
        """
        x: torch.tensor B x C x H x W
        emb: torch.tensor B x D_emb
        """

        hs = []

        h = self.input_projection(x)
        emb_bcasts = []
        for level in range(self.n_updown_levels):
            emb_bcast = emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h.shape[2], h.shape[3])
            if self.band_embedding_dim > 0:
                band_emb = self.get_band_embeddings(h.shape[2], h.device).repeat(h.shape[0], 1, 1, h.shape[3])
                emb_bcast = torch.cat((band_emb, emb_bcast), 1)
            emb_bcasts.append(emb_bcast)
            h = self.enc_blocks[level](h, emb_bcast)
            h = self.ds_layers[level](h)
            hs.append(h)
        # one more for first upsampling layer
        emb_bcast = emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h.shape[2], h.shape[3])
        if self.band_embedding_dim > 0:
                band_emb = self.get_band_embeddings(h.shape[2], h.device).repeat(h.shape[0], 1, 1, h.shape[3])
                emb_bcast = torch.cat((band_emb, emb_bcast), 1)
        emb_bcasts.append(emb_bcast)
        h = self.middle_block(h, emb_bcast)

        for level in range(self.n_updown_levels):
            h = h + hs.pop()
            emb_bcast = emb_bcasts.pop()
            h = self.dec_blocks[level](h, emb_bcast)
            h = self.us_layers[level](h)

        h = self.output_projection(h)
        return h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
        rotary_dims = 16
    ):
        super().__init__()

        #self.positional_embedding = nn.Parameter(
        #    th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        #)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        #x = x # + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class RotaryAttentionPool2d(nn.Module):
    def __init__(
            self,
            rotary_dim=32,
            attn_dim: int = None,
            embed_dim: int = None,
            num_heads: int = None,
            output_dim: int = None,
            use_input_norm: bool = False
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.gnorm = GroupNorm32(32, embed_dim)
        self.q_proj = nn.Conv2d(embed_dim, attn_dim, 1)
        self.k_proj= nn.Conv2d(embed_dim, attn_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, output_dim, 1)
        self.num_heads = num_heads
        self.pos_emb = RotaryEmbedding(
            dim=rotary_dim,
            freqs_for='pixel',
            max_freq=64)

    def forward(self, x):
        """
        x: tensor of shape b x c x h x w
        """
        if self.use_input_norm:
            x = self.gnorm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # split heads:
        _b, _dims, _height, _width = q.shape
        attn_head_dim = self.attn_dim // self.num_heads
        out_head_dim = self.output_dim // self.num_heads
        q = q.view(_b, self.num_heads, attn_head_dim, _height, _width)
        k = k.view(_b, self.num_heads, attn_head_dim, _height, _width)
        v = v.view(_b, self.num_heads, out_head_dim, _height, _width)
        q = q.permute(0,1,3,4,2)
        k = k.permute(0,1,3,4,2)
        v = v.permute(0,1,3,4,2)
        # apply rotary
        freqs = self.pos_emb.get_axial_freqs(_height, _width)

        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        # squash h and w dimensions
        q = q.view(_b, self.num_heads, _height*_width, attn_head_dim)
        k = k.view(_b, self.num_heads, _height*_width, attn_head_dim)
        v = v.view(_b, self.num_heads, _height*_width, out_head_dim)
        attn_out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous()) # b x self.num_heads x hw x out_head_dim
        attn_out = attn_out.view(_b, self.num_heads, _height, _width, out_head_dim)

        attn_out = attn_out.permute(0, 1, 4, 2, 3)
        attn_out = attn_out.reshape(_b, self.output_dim, _height, _width)
        return attn_out


class SinusoidalTemporalEmbedding(nn.Module):
    def __init__(self, n_bands, min_freq=1, max_freq=16):
        super().__init__()
        self.n_bands = n_bands
        multipliers = torch.linspace(min_freq, max_freq, n_bands).unsqueeze(0)
        self.register_buffer('multipliers', multipliers)

    def forward(self, t):
        """
        input:
            t: torch.tensor of dims B (batch) [0,1]
        output:
            t_emb: torch.tensor of dims B x 2*n_bands
        """
        sin_vals = torch.sin(t[:, None] @ self.multipliers)
        cos_vals = torch.cos(t[:, None] @ self.multipliers)
        return torch.cat((sin_vals, cos_vals), -1)
