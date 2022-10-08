import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import logging

from mesa import custom_quant
from mesa import native
from mesa import packbit

from .sparse_matrix import sparsify, unsparsify
from pdb import set_trace

# Uniform Quantization based Convolution
class conv2d_uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mask, stride, padding, dilation, groups, clip_val, level, iteration, ema_decay, quant_groups, shift):
        shape_x, mask_x, sparse_x = sparsify(x, mask, with_batch_size=False)

        # custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(shape_x, mask_x, sparse_x, weight, bias)
        ctx.hyperparameters_conv = (stride, padding, dilation, groups)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = None, None, None

        shape_x, mask_x, sparse_x, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.hyperparameters_conv

        x = unsparsify(shape_x, mask_x, sparse_x)
        # x = custom_quant.Quant.restore(ctx)
        # conv
        benchmark = True
        deterministic = True
        allow_tf32 = True
        output_mask = [True, True] #ctx.needs_input_grad[:2]
        grad_output = grad_output.to(dtype=weight.dtype)
        x = x.to(dtype=weight.dtype)
        if torch.__version__ >= "1.7":
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups,
                    benchmark, deterministic, allow_tf32, output_mask)
        else:
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups,
                    benchmark, deterministic, output_mask)
        x = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        ctx.conv_weight = None
        ctx.hyperparameters_conv = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None


class SparseConv2d(nn.Conv2d, custom_quant.Quant):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, args=None, logger=None, quant_groups=1, masker=None, act_prune=False):
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.masker = masker
        self.act_prune = act_prune
        # print("act_prune is {}".format(act_prune))
        self.tag = 'conv'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.masker is not None and self.training:
            mask = self.masker(x)
            if self.act_prune:
                # apply mask to activation in the forward
                x = x * mask
            y = conv2d_uniform.apply(x, self.weight, self.bias, mask, self.stride, self.padding, self.dilation, self.groups,
                                     self.clip_val, self.level, self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
