import torch
import torch.nn as nn
import torch.nn.functional as F

from mesa import custom_quant
from mesa import native
from mesa import packbit

from .sparse_matrix import sparsify, unsparsify


class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, quantize=False, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        shape_x, mask_x, sparse_x = sparsify(x, mask)
        if half and (not quantize):
            sparse_x = sparse_x.half()

        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(shape_x, mask_x)
        else:
            ctx.save_for_backward(shape_x, mask_x, sparse_x)

        y = F.gelu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors

        if len(tensors) == 2:
            shape_x, mask_x = tensors
            sparse_x = custom_quant.Quant.restore(ctx)
        else:
            shape_x, mask_x, sparse_x = tensors

        sparse_x = sparse_x.float()
        x = unsparsify(shape_x, mask_x, sparse_x)
        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)

        return grad_input, None, None, None, None, None, None, None, None, None


class geluMaskFree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantize=False, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        if quantize:
            custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(torch.BoolTensor([False,]))
        else:
            ctx.save_for_backward(x.half())

        y = F.gelu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if x.numel() == 1 and isinstance(x, torch.BoolTensor):
            x = custom_quant.Quant.restore(ctx)
        else:
            x = x.float()

        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)

        return grad_input, None, None, None, None, None, None, None, None


class GELUSparse(nn.GELU, custom_quant.Quant):
    def __init__(self, args=None, logger=None, quant_groups=1, masker=None, quantize=False, half=False):
        super(GELUSparse, self).__init__()
        self.masker = masker
        self.quantize = quantize
        self.half = half
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'gelu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.masker is not None and self.training:
            mask = self.masker(x)
            y = gelu.apply(x, mask, self.quantize, self.half, self.clip_val, self.level,
                           self.iteration, self.ema_decay, self.quant_groups, self.shift)
        elif self.half and self.training:
            y = geluMaskFree.apply(x, self.quantize, self.half, self.clip_val, self.level,
                                   self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.gelu(x)
        return y


if __name__ == "__main__":
    model = GELU()
    print(model)
    model.enable = True
    print(model)

    import mesa as  ms
    model = ms.GELU()
    print(model)
    model.enable = True
    print(model)
