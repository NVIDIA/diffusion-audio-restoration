# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


from torchaudio.transforms import Spectrogram, InverseSpectrogram
from torch import Tensor
import torch
import torch.nn as nn
from torch.linalg import svd as svd
from typing import List, TypeVar, Optional
from einops import rearrange
import jsonargparse

T = TypeVar("T")

import inspect
from pydoc import locate
from jsonargparse import Namespace
import importlib
from functools import partial

def instantiate_from_ns(ns: Namespace):
    if not isinstance(ns, Namespace):
        return ns

    target = getattr(ns, "class_path", None)
    if not target:
        raise ValueError("Expected 'class_path' in Namespace.")

    # Resolve the class/function from dotted path
    obj = locate(target)
    if obj is None:  # fallback
        mod, _, name = target.rpartition(".")
        if not mod:
            raise ImportError(f"Cannot import '{target}'")
        obj = getattr(importlib.import_module(mod), name)

    # kwargs (recursively) from init_args
    init_ns = getattr(ns, "init_args", None)
    kwargs = init_ns.as_dict() if isinstance(init_ns, Namespace) else (init_ns or {})

    # If it's a class, instantiate. If it's a function, return a partial.
    if inspect.isclass(obj):
        return obj(**kwargs)
    if callable(obj):
        return partial(obj, **kwargs)

    raise TypeError(f"{target} is neither a class nor a callable.")


def apply_audio_transforms(audio: torch.Tensor, transforms: List[T]):
    """
    inputs:
        audio: torch tensor of audio data. User is responsible for ensuring format matches transform function input specifications
    returns:
        audio_transformed: audio after applying series of transforms
        mask: accumulation of masks
    """
    masks = []

    for tx_fn in transforms:
        # print("type of tx_fn", type(tx_fn), tx_fn)
        if type(tx_fn) in [jsonargparse._namespace.Namespace]:
            tx_fn = instantiate_from_ns(tx_fn)
        
        output = tx_fn(audio)
        if type(output) is tuple:
            masks.append(output[1])
            audio = output[0]
        else:
            audio = output
    mask = None
    if len(masks) > 0:
        mask = torch.stack(masks).sum(0).clamp(0,1)

    return audio, mask


class ComplexSpectrogram:
    def __init__(self, n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 eps=0.000000001
                 ):
        super().__init__()
        self.eps = eps
        self.spectrogram = Spectrogram(n_fft=n_fft,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       window_fn=torch.hann_window,
                                       power=None # required for complex
                                       )

    def __call__(self, waveform: Tensor) -> Tensor:
        """
        waveform: length T tensor of waveform audio
        """
        assert len(waveform.shape) == 1, waveform.shape
        spec = self.spectrogram(waveform)
        spec = torch.view_as_real(spec).permute(2, 0, 1)
        return spec


class ComplexToMagInstPhase:
    def __call__(self, complex_spec: Tensor) -> Tensor:
        """
        inputs:
            complex_spec: 2 x H x W complex spectrogram viewed as 2-channel real values
        outputs:
            output: 3 x H x W spectrogram where channels are magnitude, cos(theta), sin(theta)
        """
        mag = torch.sqrt(complex_spec[0:1] ** 2 + complex_spec[1:2] ** 2)
        phase = torch.atan2(complex_spec[1:2],complex_spec[0:1])
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], 0)


class MagInstPhaseToComplex:
    def __call__(self, msp_spec: Tensor) -> Tensor:
        """
        inputs:
            msp_spec: magnitude + instantaneous phase representation 3 x H x W
        outputs:
            complex_spec: 2 x H x W complex spectrogram viewed as real
        """
        mag = msp_spec[:1]
        cos_theta = msp_spec[1:2]
        sin_theta = msp_spec[2:3]
        return torch.cat([mag*cos_theta, mag*sin_theta], 0)


class SVDFixMagInstPhase:
    def __call__(self, msp_spec: Tensor) -> Tensor:
        """
        cleans up the inconsistent phase predictions by using SVD to map to proper SO(2) parameters
        inputs:
            msp_spec: magnitude + instantaneous phase representation 3 x H x W
        outputs:
            msp_spec: magnitude + instantaneous phase representation 3 x H x W
        """
        mag = msp_spec[:1]
        cos_theta = msp_spec[1:2] # 1 x n_bands x t
        sin_theta = msp_spec[2:3] # 1 x n_bands x t
        top = torch.cat([cos_theta, -sin_theta], 0)
        bottom = torch.cat([sin_theta, cos_theta], 0)
        rot = torch.stack([top, bottom], 0) # 2 x 2 x n_bands x t
        rot = rearrange(rot, "r c n t -> n t r c") # move matrix to end
        U, S, Vh = svd(rot)
        new_S = S.clone()
        new_S[...,0] = 1
        new_S[...,1] = torch.det(U@Vh)
        new_rot = U @ torch.diag_embed(new_S) @ Vh
        new_cos_sin_theta = new_rot[...,:,0]
        new_cos_sin_theta = rearrange(new_cos_sin_theta, "n t r -> r n t")
        new_msp_spec = torch.cat([mag, new_cos_sin_theta], 0)

        return new_msp_spec


class InverseComplexSpectrogram:
    def __init__(self, n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 eps=0.000000001
                 ):
        super().__init__()
        self.eps = eps
        self.inv_spectrogram = InverseSpectrogram(n_fft=n_fft,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              window_fn=torch.hann_window
                                              )

    def __call__(self, spec: Tensor) -> Tensor:
        """
        spec: 2 x H X W real spectrogram tensor
        """
        assert(len(spec.shape) == 3), '{} shape not correct'.format(spec.shape)
        # move channels to end, then convert back to complex
        spec = torch.view_as_complex(spec.permute(1, 2, 0).contiguous())
        return self.inv_spectrogram(spec)


class PowerScaleSpectrogram:
    def __init__(self, power=0.5, channels=None, eps=0.000000001):
        super().__init__()
        self.eps = eps
        self.power = power
        self.channels = channels

    def __call__(self, spec: Tensor) -> Tensor:
        """
        spec: C x H x W tensor
        """

        spec_abs = spec.abs()
        scale = spec_abs ** self.power / (spec_abs + self.eps)
        if self.channels is None:
            spec = spec * scale
        else:
            inds_to_scale = torch.tensor(self.channels)
            spec = spec.clone()
            spec[inds_to_scale] = spec[inds_to_scale] * scale[inds_to_scale]
        return spec


"""
Drop or restore DCTerm, assumed to be the first FFT band
This band tends to be inconsequential, so restoring with zeros is fine
"""
class SpectrogramDropDCTerm:
    def __init__(self):
        super().__init__()

    def __call__(self, spec: Tensor) -> Tensor:
        return spec[...,1:,:]


class SpectrogramAddDCTerm:
    def __init__(self):
        super().__init__()

    def __call__(self, spec: Tensor) -> Tensor:
        dc_channel = spec[...,:1,:] * 0 # get a random slice and zero it out
        return torch.cat((dc_channel, spec), -2)


class MagInstPhaseToGriffinLim(nn.Module):
    def __init__(self, n_fft=1024,
                win_length=1024,
                hop_length=256):
        super().__init__()
        self.window = torch.hann_window(win_length)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, mag_inst_phase: Tensor) -> Tensor:
        window = self.window.to(mag_inst_phase.device)
        result = griffinlim(
            mag_inst_phase[0], 
            mag_inst_phase[2],
            mag_inst_phase[1],
            window=window,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
            n_iter=128,
            momentum=.99,
            rand_init=True,
            length=None
        )

        return result


def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")


# taken from torchaudio.functional.griffinlim
# modified to accept initialized phase
def griffinlim(
    specgram: Tensor,
    init_phase_cos: Optional[Tensor],
    init_phase_sin: Optional[Tensor],
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    n_iter: int,
    momentum: float,
    length: Optional[int],
    rand_init: bool,
) -> Tensor:
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Implementation ported from
    *librosa* :cite:`brian_mcfee-proc-scipy-2015`, *A fast Griffin-Lim algorithm* :cite:`6701851`
    and *Signal estimation from modified short-time Fourier transform* :cite:`1172092`.

    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension `(..., freq, frames)`
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.

    Returns:
        Tensor: waveform of `(..., time)`, where time equals the ``length`` parameter if given.
    """
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in range [0, 1). Found: {}".format(momentum))

    momentum = momentum / (1 + momentum)

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    specgram = specgram.pow(1 / power)

    # initialize the phase
    if rand_init:
        angles = torch.rand(specgram.size(), dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
    else:
        # angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
        angles = torch.complex(init_phase_cos, init_phase_sin).to(_get_complex_dtype(specgram.dtype)).to(specgram.device)

    # And initialize the previous iterate to 0
    tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = torch.istft(
            specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
        )

        # Rebuild the spectrogram
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum)
        angles = angles.div(angles.abs().add(1e-16))

        # Store the previous iterate
        tprev = rebuilt

    # Return the final phase estimates
    waveform = torch.istft(
        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
    )

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform
