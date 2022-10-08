import torch
import torch.nn as nn
import torch.nn.functional as F

from mesa import custom_quant
from mesa import native
from mesa import packbit

from .sparse_matrix import sparsify, unsparsify

from pdb import set_trace

class softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, masker, quantize, half, dim,
                clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None,
                clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None):

        y = F.softmax(x, dim)
        mask = masker(y)

        if quantize:
            shape_y, mask_y, sparse_y = sparsify(y, mask)
            ctx.save_for_backward(shape_y, mask_y)
            custom_quant.Quant.forward(ctx, sparse_y, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2)
        else:
            shape_y, mask_y, sparse_y = sparsify(y, mask)

            if half:
                sparse_y = sparse_y.half()

            ctx.save_for_backward(shape_y, mask_y, sparse_y)

        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_in):
        tensors = ctx.saved_tensors

        if len(tensors) == 2:
            shape_y, mask_y = tensors
            sparse_y = custom_quant.Quant.restore(ctx)
        else:
            shape_y, mask_y, sparse_y = tensors

        y = unsparsify(shape_y, mask_y, sparse_y)

        grad_out = grad_in * y - ((grad_in.unsqueeze(-2) @ y.unsqueeze(-1)) @ y.unsqueeze(-2)).squeeze(-2)
        # if y.is_cuda:
        #     grad_out = native.softmax_backward_cuda(grad_in, y, ctx.dim, y)
        # else:
        #     grad_out = native.softmax_backward_cpu(grad_in, y, ctx.dim, y)

        return grad_out, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class SoftmaxSparse(nn.Softmax):
    def __init__(self, dim=None, args=None, logger=None, quant_groups=1, masker=None, quantize=False, half=False):
        super(SoftmaxSparse, self).__init__(dim=dim)
        self.quant1 = custom_quant.quantization(tag='softmax-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='softmax-2', quant_groups=quant_groups)
        self.masker = masker
        self.quantize = quantize
        self.half = half
        self.tag = 'softmax'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x):
        if self.masker is not None and self.training:
            y = softmax.apply(x, self.masker, self.quantize, self.half, self.dim,
                              self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift,
                              self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift)
        else:
            y = F.softmax(x, self.dim)
        return y


