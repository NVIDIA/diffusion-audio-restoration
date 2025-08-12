# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
import librosa


def plot_spec_to_numpy(spectrogram, title='', sr=48000, hop_length=512, info=None, vmin=None, vmax=None, cmap='brg'):
    fig, ax = plt.subplots(figsize=(6, 4))
    spec_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    img = librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='frames', y_axis='linear', ax=ax)

    fig.colorbar(img, ax=ax)
    fig.tight_layout()

    fig.canvas.draw()
    fig.show()
    numpy_fig = mplfig_to_npimage(fig)

    return numpy_fig


def plot_phase_to_numpy(phase, title='', sr=48000, hop_length=512, info=None, vmin=-np.pi, vmax=np.pi, cmap='hsv'):
    fig, ax = plt.subplots(figsize=(6, 4))
    phase_np = phase.numpy()
    
    img = librosa.display.specshow(phase_np, sr=sr, hop_length=hop_length, x_axis='frames', y_axis='linear', cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f rad')
    cbar.set_label('Phase (radians)')

    ax.set_title(title if title else 'Spectrogram Phase')
    fig.tight_layout()

    fig.canvas.draw()
    fig.show()
    numpy_fig = mplfig_to_npimage(fig)
    matplotlib.pyplot.close(fig)
    return numpy_fig
