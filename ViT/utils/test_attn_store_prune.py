# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from pdb import set_trace
import sys
sys.path.append(".")

from models.modeling_attn_store_prune import SoftmaxActivationPrune, AttentionStoreActivationPrune, MlpActivationPrune
from models.modeling import Attention, Mlp


def testSoftMax():
    A = torch.rand(1, 4, 32, 32)
    A.requires_grad = True

    # origin softmax
    A_softmax_ori = Softmax(dim=-1)(A)
    A_softmax_ori.sum().backward()

    A_grad_ori = A.grad

    # our softmax
    A.grad = None

    # when prune ratio is 0, the two should be equal
    A_softmax_our, _ = SoftmaxActivationPrune.apply(A, 0)
    A_softmax_our.sum().backward()

    A_grad_our = A.grad

    print("activation dist is {}".format(torch.norm(A_softmax_ori - A_softmax_our)))
    print("grad dist is {}".format(torch.norm(A_grad_ori - A_grad_our)))


class Config(object):
    def __init__(self):
        self.hidden_size = 32

        self.transformer = dict()
        self.transformer["attention_dropout_rate"] = 0
        self.transformer["num_heads"] = 4

        self.transformer["mlp_dim"] = 64
        self.transformer["dropout_rate"] = 0


def testAttnStoreActivationPrune():
    config = Config()

    attn_origin = Attention(config, False)
    attn_our = AttentionStoreActivationPrune(config, False, prune_ratio_attn_mat_store=0, prune_ratio_act_store=0)
    attn_our.load_state_dict(attn_origin.state_dict())

    input = torch.rand(2, 10, 32)
    input.requires_grad = True

    attn_origin_out = attn_origin(input)
    attn_origin_out[0].sum().backward()
    input_grad_ori = input.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    attn_our_out = attn_our(input)
    attn_our_out[0].sum().backward()

    input_grad_our = input.grad

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(attn_our_out[0] - attn_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))


def testMlpStoreActivationPrune():
    config = Config()

    mlp_origin = Mlp(config)
    mlp_our = MlpActivationPrune(config, prune_ratio_act_store=0)
    mlp_our.load_state_dict(mlp_origin.state_dict())

    input = torch.rand(2, 10, 32)
    input.requires_grad = True

    mlp_origin_out = mlp_origin(input)
    mlp_origin_out[0].sum().backward()
    input_grad_ori = input.grad
    fc1_grad_origin = mlp_origin.fc1.weight.grad
    fc2_grad_origin = mlp_origin.fc2.weight.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    mlp_our_out = mlp_our(input)
    mlp_our_out[0].sum().backward()
    fc1_grad_our = mlp_our.fc1.weight.grad
    fc2_grad_our = mlp_our.fc2.weight.grad

    input_grad_our = input.grad

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(mlp_our_out[0] - mlp_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("input grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))
    print("fc1 grad dist is {}".format(torch.norm(fc1_grad_origin - fc1_grad_our)))
    print("fc2 grad dist is {}".format(torch.norm(fc2_grad_origin - fc2_grad_our)))


class ConfigMemoryTest(object):
    def __init__(self):
        self.hidden_size = 384

        self.transformer = dict()
        self.transformer["attention_dropout_rate"] = 0
        self.transformer["num_heads"] = 12

        self.transformer["mlp_dim"] = 384
        self.transformer["dropout_rate"] = 0

def testMemoryMlpStoreActivationPrune():
    config = ConfigMemoryTest()

    prune_ratio = 0.8

    model = nn.Sequential(*[MlpActivationPrune(config, prune_ratio_act_store=prune_ratio) for _ in range(10)])

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    model = model.cuda()
    input = torch.rand(64, 196, 384).cuda()


    mlp_origin_out = model(input)
    mlp_origin_out[0].sum().backward()


    print("############ prune ratio of {} #############".format(prune_ratio))
    MB = 1024.0 * 1024.0
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))


def testMemoryMlp():
    config = ConfigMemoryTest()

    model = nn.Sequential(*[Mlp(config) for _ in range(10)])

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    model = model.cuda()
    input = torch.rand(64, 196, 384).cuda()
    MB = 1024.0 * 1024.0
    print("input usage is {:.1f} MB".format(input.element_size() * input.nelement() / MB))

    mlp_origin_out = model(input)
    mlp_origin_out[0].sum().backward()

    print("############ standard mlp #############")
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))


def testMemoryWeightNoGradMlp():
    config = ConfigMemoryTest()

    model = nn.Sequential(*[Mlp(config) for _ in range(10)])
    model = model.cuda()
    input = torch.rand(64, 196, 384).cuda()

    mlp_origin_out = model(input)

    for name, param in model.named_parameters():
        if "weight" in name:
            print(name)
            param.requires_grad = False

    mlp_origin_out[0].sum().backward()


    print("############ weight no gradient mlp #############")
    MB = 1024.0 * 1024.0
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))


def testMemoryMesaMlp():
    config = ConfigMemoryTest()

    from model_mesa.modeling_mesa import Mlp as MlpMesa
    import mesa as ms
    model = nn.Sequential(*[MlpMesa(config) for _ in range(10)])

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    for name, module in model.named_modules():
        module.name = name
    ms.policy.deploy_on_init(model, 'model_mesa/policy_tiny-8bit.txt', verbose=print, override_verbose=False)
    model = model.cuda()
    input = torch.rand(64, 196, 384).cuda()

    mlp_origin_out = model(input)

    mlp_origin_out[0].sum().backward()


    print("############ weight no gradient mlp #############")
    MB = 1024.0 * 1024.0
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))


if __name__ == "__main__":
    # testSoftMax()
    # testMlpStoreActivationPrune()

    # testMemoryMlpStoreActivationPrune()
    # testMemoryMlp()
    # testMemoryWeightNoGradMlp()
    testMemoryMesaMlp()
