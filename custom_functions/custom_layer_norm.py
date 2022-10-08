import torch
import torch.nn as nn
import torch.nn.functional as F


from mesa import custom_quant
from mesa import native
from mesa import packbit

from .sparse_matrix import sparsify, unsparsify

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, mask, quantize, half, eps, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        if x.dtype != weight.data.dtype:
            x = x.to(dtype=weight.data.dtype)

        shape_x, mask_x, sparse_x = sparsify(x, mask)

        if half and (not quantize):
            sparse_x = sparse_x.half()

        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(shape_x, mask_x)
        else:
            ctx.save_for_backward(shape_x, mask_x, sparse_x)

        if torch.__version__ >= "1.8":
            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, normalized_shape, weight, bias, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, normalized_shape, weight, bias, eps)

            ctx.layer_norm_parameters = (mean, rstd, weight, bias, normalized_shape)
        else:
            N = 1
            if isinstance(normalized_shape, int):
                N = normalized_shape
            elif isinstance(normalized_shape, (list, tuple)):
                for i in normalized_shape:
                    N *= i
            else:
                raise RuntimeError("type of normalized_shape".format(type(normalized_shape)))
            M = x.nelement() // N

            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, weight, bias, M, N, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, weight, bias, M, N, eps)

            ctx.layer_norm_parameters = (mean, rstd, weight, M, N)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors

        if len(tensors) == 2:
            shape_x, mask_x = tensors
            grad_output = grad_output.contiguous()
            sparse_x = custom_quant.Quant.restore(ctx)
        else:
            shape_x, mask_x, sparse_x = tensors

        sparse_x = sparse_x.float()
        x = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=True)

        grad_input = grad_weight = grad_bias = None

        output_mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]

        if torch.__version__ >= "1.8":
            mean, rstd, weight, bias, normalized_shape = ctx.layer_norm_parameters
            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = \
                    native.layer_norm_backward_cuda(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
            else:
                grad_input, grad_weight, grad_bias = \
                    native.layer_norm_backward_cpu(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
        else:
            mean, rstd, weight, M, N = ctx.layer_norm_parameters

            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cuda(grad_output, x, mean, rstd, weight, M, N, output_mask)
            else:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cpu(grad_output, x, mean, rstd, weight, M, N, output_mask)
        ctx.layer_norm_parameters = None

        return grad_input, None, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None


class LayerNormSparse(nn.LayerNorm, custom_quant.Quant):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, args=None, logger=None, quant_groups=1,
                 masker=None, quantize=False, half=False, backrazor_bits=32):
        super(LayerNormSparse, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'layernorm'
        self.masker = masker
        self.quantize = quantize
        self.half = half

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.masker is not None and self.training:
            mask = self.masker(x)
            y = layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, mask, self.quantize, self.half, self.eps,
                                 self.clip_val, self.level,
                                 self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return y