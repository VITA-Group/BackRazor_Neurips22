import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

from mesa import custom_quant
from mesa import native
from mesa import packbit

from .sparse_matrix import sparsify, unsparsify

# design for Softmax(A)v
class softmax_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, masker, quantize, half, dim,
                clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None,
                clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None):

        input1_sm = F.softmax(input1, dim)
        ctx.dim = dim

        mask1 = masker(input1_sm)
        mask2 = masker(input2)

        shape_x_1, mask_x_1, sparse_x_1 = sparsify(input1_sm, mask1)
        shape_x_2, mask_x_2, sparse_x_2 = sparsify(input2, mask2)

        if half and (not quantize):
            sparse_x_1 = sparse_x_1.half()
            sparse_x_2 = sparse_x_2.half()

        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x_1, clip_val1, level1, iteration1, ema_decay1, quant_groups1, shift1, '_1')
            custom_quant.Quant.forward(ctx, sparse_x_2, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2, '_2')

            ctx.save_for_backward(shape_x_1, shape_x_2, mask_x_1, mask_x_2)
        else:
            ctx.save_for_backward(shape_x_1, shape_x_2, mask_x_1, mask_x_2, sparse_x_1, sparse_x_2)

        output = input1_sm.matmul(input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None

        tensors = ctx.saved_tensors
        if len(tensors) == 4:
            shape_x_1, shape_x_2, mask_x_1, mask_x_2 = tensors

            sparse_x_1 = custom_quant.Quant.restore(ctx, '_1')
            sparse_x_2 = custom_quant.Quant.restore(ctx, '_2')
        else:
            shape_x_1, shape_x_2, mask_x_1, mask_x_2, sparse_x_1, sparse_x_2 = tensors

        sparse_x_1 = sparse_x_1.float()
        sparse_x_2 = sparse_x_2.float()

        input1_sm = unsparsify(shape_x_1, mask_x_1, sparse_x_1)
        input2 = unsparsify(shape_x_2, mask_x_2, sparse_x_2)

        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.transpose(-2, -1).to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1_sm.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)

        if input1_sm.is_cuda:
            grad_input1 = native.softmax_backward_cuda(grad_input1, input1_sm, ctx.dim, input1_sm)
        else:
            grad_input1 = native.softmax_backward_cpu(grad_input1, input1_sm, ctx.dim, input1_sm)

        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class SoftmaxMatMulSparse(nn.Module):
    def __init__(self, args=None, logger=None, quant_groups=1, masker=None, quantize=False, half=False, dim=-1):
        super(SoftmaxMatMulSparse, self).__init__()
        self.quant1 = custom_quant.quantization(tag='matmul-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='matmul-2', quant_groups=quant_groups)
        self.quantize = quantize
        self.half = half
        self.masker = masker
        self.dim = dim
        self.tag = 'matmul'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x1, x2):
        if self.masker is not None or self.training:
            y = softmax_matmul.apply(x1, x2, self.masker, self.quantize, self.half, self.dim,
                                     self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift,
                                     self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift)
        else:
            y = torch.matmul(x1, x2)
        return y


if __name__ == "__main__":
    model = MatMul()
    print(model)

    model.quant1.enable = True
    model.quant2.enable = True
    print(model)

