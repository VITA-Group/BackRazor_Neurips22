# heavily borrow from

import copy
import torch
import torch.nn as nn
import mesa as ms
# from ofa.utils import Hswish, Hsigmoid, MyConv2d

# from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenet import InvertedResidual

from ViT.models.modeling import Attention, Mlp
from ViT.models.modeling_new_prune import AttentionActPrune, MlpActPrune

from pdb import set_trace
from custom_functions.custom_fc import LinearSparse
from custom_functions.custom_softmax import SoftmaxSparse
from custom_functions.custom_gelu import GELUSparse
from custom_functions.custom_layer_norm import LayerNormSparse


__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
	frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

	trainable_param_size = 0
	frozen_param_size = 0
	for p in net.parameters():
		if p.requires_grad:
			trainable_param_size += trainable_param_bits / 8 * p.numel()
		else:
			frozen_param_size += frozen_param_bits / 8 * p.numel()
	model_size = trainable_param_size + frozen_param_size
	if print_log:
		print('Total: %d' % model_size,
		      '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
		      '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
	# Byte
	return model_size


def is_leaf(m_):
	return len(list(m_.children())) == 0 or isinstance(m_, LinearSparse) or isinstance(m_, SoftmaxSparse) or \
		   isinstance(m_, GELUSparse) or isinstance(m_, LayerNormSparse) or \
		   (len(list(m_.children())) == 1 and isinstance(next(m_.children()), nn.Identity))


def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32, head_only=False):
	act_byte = activation_bits / 8
	model = copy.deepcopy(net)

	# noinspection PyArgumentList
	def count_convNd(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

	# noinspection PyArgumentList
	def count_linear(m, x, y):
		# print("count_linear")
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		if isinstance(m, LinearSparse) and m.masker is not None:
			if m.half:
				ratio = 0.5
			else:
				ratio = 1

			mask = m.masker(x[0])
			# print("mlp density is {}".format(mask.float().mean().cpu()))
			m.grad_activations *= mask.float().mean().cpu() * ratio
			if m.quantize:
				m.grad_activations *= 0.25
			m.grad_activations += (mask.numel() / 8)

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_quantized_linear(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			if m.enable:
				ratio = 0.25
			else:
				ratio = 1.0
			# print("count_quantized_linear, enable {}".format(m.enable))
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte * ratio])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_bn(m, x, _):
		# print("count LN")
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		if isinstance(m, LayerNormSparse) and m.masker is not None:
			mask = m.masker(x[0])
			# print("layer norm density is {}".format(mask.float().mean().cpu()))
			m.grad_activations *= mask.float().mean().cpu()
			if m.quantize:
				m.grad_activations *= 0.25
			m.grad_activations += (mask.numel() / 8)

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_quantized_bn(m, x, _):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			if m.enable:
				ratio = 0.25
			else:
				ratio = 1.0
			# print("count quantized LN, enable {}".format(m.enable))
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte * ratio])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_relu(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_smooth_act(m, x, _):
		# print("count gelu")
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	def add_hooks(m_):
		# if isinstance(m_, nn.GELU):
		# 	set_trace()

		if not is_leaf(m_):
			return

		m_.register_buffer('grad_activations', torch.zeros(1))
		m_.register_buffer('tmp_activations', torch.zeros(1))

		if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d, ]:
			fn = count_convNd
		elif type(m_) in [nn.Linear, LinearSparse]:
			fn = count_linear
		elif type(m_) in [ms.Linear]:
			fn = count_quantized_linear
		elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, LayerNormSparse]:
			fn = count_bn
		elif type(m_) in [ms.LayerNorm]:
			fn = count_quantized_bn
		elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
			fn = count_relu
		elif type(m_) in [nn.Sigmoid, nn.Tanh]:
			fn = count_smooth_act
		else:
			fn = None

		if fn is not None:
			_handler = m_.register_forward_hook(fn)

	model.train()
	model.apply(add_hooks)

	x = torch.randn(input_size).to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x)

	memory_info_dict = {
		'peak_activation_size': torch.zeros(1),
		'grad_activation_size': torch.zeros(1),
		'residual_size': torch.zeros(1),
	}

	for m in model.modules():
		if is_leaf(m):
			def new_forward(_module):
				def lambda_forward(*args, **kwargs):
					current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
					                   memory_info_dict['residual_size']
					memory_info_dict['peak_activation_size'] = max(
						current_act_size, memory_info_dict['peak_activation_size']
					)
					memory_info_dict['grad_activation_size'] += _module.grad_activations
					return _module.old_forward(*args, **kwargs)

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		if isinstance(m, Attention) or isinstance(m, AttentionActPrune):
			def new_forward(_module):
				def lambda_forward(_x):
					# print("count Attention")
					memory_info_dict['residual_size'] = _x[0].numel() * act_byte
					# save key, query, value
					# print("_x.shape is {}".format(_x.shape))
					n_tokens = _x.shape[1]
					# set_trace()

					if isinstance(m, AttentionActPrune) and _module.query.enable:
						backward_act_byte = 1
					elif isinstance(_module, AttentionActPrune) and _module.query.quantize:
						backward_act_byte = 1
					else:
						backward_act_byte = act_byte

					# print("attention, backward_act_byte is {}".format(backward_act_byte))
					if isinstance(_module, AttentionActPrune):
						attn_prune_ratio = _module.masker.prune_ratio
						# print("attn_prune_ratio is {}".format(attn_prune_ratio))
						# attn matrix
						if _module.query.half:
							assert backward_act_byte == 4
							ratio = 0.5
						else:
							ratio = 1

						# attn_matrix
						memory_info_dict['grad_activation_size'] += _module.num_attention_heads * n_tokens * n_tokens * (backward_act_byte) * (1 - attn_prune_ratio) * ratio
						memory_info_dict['grad_activation_size'] += _module.num_attention_heads * n_tokens * n_tokens * 1/8
						# key, query, value
						memory_info_dict['grad_activation_size'] += 3 * _module.all_head_size * n_tokens * (backward_act_byte) * (1 - attn_prune_ratio) * ratio
						memory_info_dict['grad_activation_size'] += 3 * _module.all_head_size * n_tokens * 1/8

					else:
						# attn_matrix
						memory_info_dict['grad_activation_size'] += _module.num_attention_heads * n_tokens * n_tokens * backward_act_byte
						memory_info_dict['grad_activation_size'] += 3 * _module.all_head_size * n_tokens * backward_act_byte

					if head_only:
						memory_info_dict['grad_activation_size'] *= 0

					result = _module.old_forward(_x)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		if isinstance(m, Mlp) or isinstance(m, MlpActPrune):
			def new_forward(_module):
				def lambda_forward(_x):
					# print("count Mlp")
					memory_info_dict['residual_size'] = _x[0].numel() * act_byte

					# gelu function
					# print("_x.shape is {}".format(_x.shape))
					n_tokens = _x.shape[1]
					# set_trace()

					if isinstance(m, MlpActPrune) and _module.act_fn.enable:
						backward_act_byte = 1
					elif isinstance(_module, MlpActPrune) and _module.act_fn.quantize:
						backward_act_byte = 1
					else:
						backward_act_byte = act_byte

					# print("mlp, backward_act_byte is {}".format(backward_act_byte))

					if isinstance(_module, MlpActPrune):
						# if isinstance(_module.act_fn, ms.GELU):
						if _module.act_fn.masker is not None:
							ratio = _module.act_fn.masker.prune_ratio
							memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens / 8
						else:
							ratio = 1

						if _module.act_fn.half and backward_act_byte == 4:
							ratio *= 0.5

						memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens * backward_act_byte * ratio
					else:
						memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens * backward_act_byte

					if head_only:
						memory_info_dict['grad_activation_size'] *= 0

					result = _module.old_forward(_x)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

	with torch.no_grad():
		model(x)

	return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=True, head_only=False,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
	param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
	activation_size, grad_activation_size = count_activation_size(net, input_size, require_backward, activation_bits, head_only)

	MB = 1024 * 1024
	print("grad activation size is {:.1f} MB".format(grad_activation_size / MB))
	memory_cost = activation_size * batch_size + param_size
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}

