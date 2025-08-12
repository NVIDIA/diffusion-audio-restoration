# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from torch import Tensor
from torchaudio.transforms import Spectrogram

def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class SequenceLength:
    """Data structure for storing sequence lengths
    """
    def __init__(self, lengths):
        self.lengths = lengths.long()
        self.mask = get_mask_from_lengths(lengths)


def average_key_value(dict_list, key):
    """
    Calculate the average value for a given key in a list of dictionaries.

    Parameters:
    dict_list (list): List of dictionaries
    key (str): The key whose values need to be averaged

    Returns:
    float: The average value
    """
    if not dict_list:
        return 0

    total = sum(d[key] for d in dict_list if key in d)
    count = sum(1 for d in dict_list if key in d)

    return total / count if count != 0 else 0


def find_middle_of_zero_segments(binary_array: torch.Tensor) -> torch.Tensor:
    """
    Find the middle indices of all continuous segments of zeros in a binary array.

    Args:
        binary_array (torch.Tensor): A 1D binary tensor containing 0s and 1s.

    Returns:
        torch.Tensor: A tensor of middle indices for each segment of zeros.
    """
    if not torch.is_tensor(binary_array) or binary_array.ndim != 1:
        raise ValueError("Input must be a 1D tensor.")

    # Find transitions
    diff = torch.diff(binary_array, prepend=torch.tensor([1], dtype=binary_array.dtype).to(binary_array.device))

    # Find start and end indices of zero segments
    start_indices = (diff == -1).nonzero(as_tuple=True)[0]
    end_indices = (diff == 1).nonzero(as_tuple=True)[0] - 1

    # Handle edge case: if array ends with a zero segment
    if binary_array[-1] == 0:
        end_indices = torch.cat([end_indices, torch.tensor([len(binary_array) - 1]).to(binary_array.device)])

    # Compute the middle indices of zero segments
    middle_indices = ((start_indices + end_indices) / 2).int()

    return middle_indices
