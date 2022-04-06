import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math


class HighwayLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=-2):
        self.gate = nn.Linear(input_size, output_size)
        self.bias = bias
    def forward(self, transformed, input):
        transform_gate = torch.sigmoid(self.gate(input) + self.bias)
        carry_gate = 1 - transform_gate
        return transform_gate * transformed + carry_gate * input


def sequential_neural_network(dimensions, activation, dropout=None):
    from itertools import zip_longest
    seq = []
    for input_dim, output_dim in zip_longest(dimensions[:-1], dimensions[1:]):
        seq.append(nn.Linear(input_dim, output_dim))
        seq.append(activation())
        if dropout is not None:
            seq.append(nn.Dropout(dropout))
    if type(seq[-1]) == nn.Dropout:
        seq.pop()
    return nn.Sequential(*seq)