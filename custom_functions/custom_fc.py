import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import logging

from mesa import custom_quant
from mesa import native

from .sparse_matrix import sparsify, unsparsify
from pdb import set_trace

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, mask=None, quantize=True, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        shape_x, mask_x, sparse_x = sparsify(x, mask, with_batch_size=False)

        if half and (not quantize):
            sparse_x = sparse_x.half()

        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(weight, bias, shape_x, mask_x)
        else:
            ctx.save_for_backward(weight, bias, shape_x, mask_x, sparse_x)

        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None

        tensors = ctx.saved_tensors

        if len(tensors) == 5:
            weight, bias, shape_x, mask_x, sparse_x = tensors
        else:
            weight, bias, shape_x, mask_x = tensors
            sparse_x = custom_quant.Quant.restore(ctx)

        sparse_x = sparse_x.float()
        input = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=False)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None


class LinearSparse(nn.Linear, custom_quant.Quant):
    def __init__(self, in_features, out_features, bias=True, args=None, logger=None, quant_groups=1, masker=None,
                 quantize=True, half=False, act_prune=False):
        super(LinearSparse, self).__init__(in_features, out_features, bias=bias)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.masker = masker
        self.quantize = quantize
        self.act_prune = act_prune
        self.half = half
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print("type(x) is {}".format(type(x)))
        if self.masker is not None and self.training:
            mask = self.masker(x)
            # print("mask sum is {}".format((~mask).sum()))
            if self.act_prune:
                x = x * mask
            y = linear.apply(x, self.weight, self.bias, mask, self.quantize, self.half, self.clip_val, self.level,
                             self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y


if __name__ == "__main__":
    model = LinearSparse(100, 100)
    print(model)
    model.enable = True
    print(model)

