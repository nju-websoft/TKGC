import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from daem.models.nn import SentenceConv


class MatchingDatasetContainer():
    def __init__(self, dataset):
        self.vocabs = dataset.vocabs[dataset.all_text_fields[0]]
        self.columns = dataset.canonical_text_fields

    def _example2tensor(self, example, part, delta=0):
        column = []
        token = []
        for ci, cn in enumerate(self.columns):
            column += [ci + delta for t in getattr(example, part + cn)]
            token += [self.vocabs.stoi[t] for t in getattr(example, part + cn)]
        return column, token

    def split(self, examples, batch_size=120, device='cpu'):
        output = []
        left_tensors = [self._example2tensor(e, 'left_') for e in examples]
        right_tensors = [self._example2tensor(
            e, 'right_', delta=len(self.columns)) for e in examples]
        label_tensor = torch.LongTensor([e.label for e in examples]).to(device)
        left_size_tensor = torch.LongTensor(
            [len(t[0]) for t in left_tensors]).to(device)
        right_size_tensor = torch.LongTensor(
            [len(t[0]) for t in right_tensors]).to(device)

        for i in range(0, label_tensor.shape[0], batch_size):
            begin = i
            end = min(label_tensor.shape[0], i + batch_size)
            batch = {'left': {}, 'right': {}}
            label = label_tensor[begin:end]
            batch['left']['size'] = left_size_tensor[begin:end]
            batch['right']['size'] = right_size_tensor[begin:end]
            uu = [torch.LongTensor(t[1]) for t in left_tensors[begin:end]]
            batch['left']['attr'] = pad_sequence(
                [torch.LongTensor(t[0]) for t in left_tensors[begin:end]]).to(device)
            batch['left']['token'] = pad_sequence(
                [torch.LongTensor(t[1]) for t in left_tensors[begin:end]]).to(device)
            batch['right']['attr'] = pad_sequence(
                [torch.LongTensor(t[0]) for t in right_tensors[begin:end]]).to(device)
            batch['right']['token'] = pad_sequence(
                [torch.LongTensor(t[1]) for t in right_tensors[begin:end]]).to(device)
            output.append((batch, label))
        return output


class Alignment(nn.Module):
    def __init__(self, word_embeddings, attr_num, dense_hidden=60, attention_topk=3, attr_dim=50):
        super(Alignment, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        self.attr_embeddings = torch.nn.Embedding.from_pretrained(torch.randn(attr_num, attr_dim) * 0.1, freeze=False)
        self.token_dim = self.embeddings.embedding_dim + attr_dim
        self.topk = attention_topk
        self.conv1 = SentenceConv(1, 100, self.token_dim)
        self.conv2 = SentenceConv(2, 100, self.token_dim)
        self.conv3 = SentenceConv(3, 100, self.token_dim)
        self.dense = nn.Sequential(
            nn.Linear(100 * 6, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 2),
        )
    def forward(self, batch):
        left, right = batch['left'], batch['right']
        l_lengths, l_tokens, l_attrs = left['size'], left['token'], left['attr']
        r_lengths, r_tokens, r_attrs = right['size'], right['token'], right['attr']
        device = l_lengths.device

        l_mask = pad_sequence([torch.ones(k) for k in l_lengths]).to(device)
        r_mask = pad_sequence([torch.ones(k) for k in r_lengths]).to(device)
        
        l_vectors = torch.cat([self.attr_embeddings(l_attrs), self.embeddings(l_tokens)], 2)
        r_vectors = torch.cat([self.attr_embeddings(r_attrs), self.embeddings(r_tokens)], 2)

        attention = torch.bmm(l_vectors.transpose(0, 1), r_vectors.transpose(0, 1).transpose(1, 2))
        attention = attention * l_mask.T.unsqueeze(2) * r_mask.T.unsqueeze(1)
        
        attention_r = torch.nn.functional.softmax(attention - 10 * (1 - l_mask.T.unsqueeze(2)), dim=1)
        attention_r = attention_r * l_mask.T.unsqueeze(2)
        attention_r = attention_r / (attention_r.sum(dim=1, keepdim=True) + 1e-13)
        
        attention_l = torch.nn.functional.softmax(attention - 10 * (1 - r_mask.T.unsqueeze(1)), dim=2)
        attention_l = attention_l * r_mask.T.unsqueeze(1)
        attention_l = attention_l / (attention_l.sum(dim=2, keepdim=True) + 1e-13)
        
        att = attention_l
        l_mean = att.sum(2) / r_lengths.unsqueeze(1)
        l_w = (att * att).sum(2) / r_lengths.unsqueeze(1) - l_mean * l_mean
        l_w = l_w / l_mean.clamp_min(0.001)

        l_a, l_i = torch.topk(att, self.topk, 2)
        l_i = l_i * l_mask.T.int().unsqueeze(2)
        l_a = l_a / l_a.sum(2, keepdim=True).clamp_min(0.001)

        l_counter = torch.zeros(l_vectors.shape, device=device)
        for j in range(0, self.topk):
            indices = l_i[:, :, j].transpose(0, 1).unsqueeze(2).expand(-1, -1, self.token_dim)
            l_counter += l_a[:, :, j].T.unsqueeze(2) * r_vectors.gather(0, indices)
        l_c = l_w.T.unsqueeze(2) * (l_vectors - l_counter).abs() * l_mask.unsqueeze(2)
        
        att = attention_r
        r_mean = att.sum(1) / l_lengths.unsqueeze(1)
        r_w = (att * att).sum(1) / l_lengths.unsqueeze(1) - r_mean * r_mean
        r_w = r_w / r_mean.clamp_min(0.001)

        r_a, r_i = torch.topk(att, self.topk, 1)
        r_i = r_i * r_mask.T.int().unsqueeze(1)
        r_a = r_a / r_a.sum(1, keepdim=True).clamp_min(0.001)

        r_counter = torch.zeros(r_vectors.shape, device=device)
        for j in range(0, self.topk):
            indices = r_i[:, j, :].T.unsqueeze(2).expand(-1, -1, self.token_dim)
            r_counter += r_a[:, j, :].T.unsqueeze(2) * l_vectors.gather(0, indices)
        r_c = r_w.T.unsqueeze(2) * (r_vectors - r_counter).abs() * r_mask.unsqueeze(2)
        
        return self.dense(torch.cat([
            self.conv1(l_c, l_mask), self.conv2(l_c, l_mask), self.conv3(l_c, l_mask),
            self.conv1(r_c, r_mask), self.conv2(r_c, r_mask), self.conv3(r_c, r_mask)], 1))