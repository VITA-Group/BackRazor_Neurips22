import torch
import numpy as np

from pdb import set_trace

class Masker(object):
    def __init__(self, prune_ratio):
        self.prune_ratio = prune_ratio

    @torch.no_grad()
    def __call__(self, activation):
        num_small = int(np.clip(activation[0].numel() * self.prune_ratio, 1, activation[0].numel()))
        activation_mag = torch.abs(activation)
        threshold, _ = torch.kthvalue(activation_mag.flatten(1), num_small)
        while len(threshold.shape) < len(activation_mag.shape):
            threshold = threshold.unsqueeze(-1)
        mask = activation_mag >= threshold

        # print("mask density is {}".format(mask.float().mean()))
        # idle mask
        # mask = torch.ones_like(activation).to(torch.bool)

        return mask
