import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math


class ResidualLayer(nn.Module):
    def forward(self, transformed, input):
        return transformed + input


def residual(transformed, input):
    return transformed + input


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)
