import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from .neural import sequential_neural_network
import math


class SentenceConv(nn.Module):
    def __init__(self, ngram=3, duplicate=100, dim=100, pooling='sum'):
        super(SentenceConv, self).__init__()
        self.ngram = ngram
        self.weights = nn.ModuleList([nn.Linear(dim, duplicate, bias=False) for i in range(self.ngram)])
        self.bias = nn.Parameter(torch.zeros((1, 1, duplicate)))
        assert pooling in ('max', 'avg', 'min', 'sum')
        self.pooling = pooling

    def forward(self, input, mask):
        max_len, _ = input.shape[:2]
        mask = mask[self.ngram-1:]
        
        tail = max_len - self.ngram + 1
        out = self.weights[0](input[0:tail])
        for j in range(1, self.ngram):
            out += self.weights[j](input[j:tail+j])
        out = (out + self.bias) * mask.unsqueeze(2)
        if self.pooling == 'max':
            out = out - (1 - mask.unsqueeze(2)) * 1e5
            return out.max(0)[0]
        elif self.pooling == 'avg':
            out = out
            print(out.sum(0).shape, mask.sum(0, keepdim=True).shape)
            return out.sum(0) / mask.sum(0, keepdim=True).T
        elif self.pooling == 'min':
            out = out + (1 - mask.unsqueeze(2)) * 1e5
            return out.min(0)[0]
        else: # elif self.pooling == 'sum'
            return out.sum(0)


class SentenceDifference(nn.Module):
    def __init__(self, max_length=128):
        super(SentenceDifference, self).__init__()
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_length + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)
    def forward(self, left_vectors, left_lengths, right_vectors, right_lengths):
        left_mask = self.mask_matrix[left_lengths, :left_vectors.shape[0]]
        right_mask = self.mask_matrix[right_lengths, :right_vectors.shape[0]]
        attention = torch.bmm(left_vectors.transpose(0, 1), right_vectors.transpose(0, 1).transpose(1, 2))
        attention = attention * left_mask.unsqueeze(2) * right_mask.unsqueeze(1)

        indices = torch.argmax(attention, 2).transpose(0, 1).unsqueeze(2).expand(-1, -1, left_vectors.shape[2])
        left_difference = left_vectors - right_vectors.gather(0, indices)
        left_difference *= left_mask.T.unsqueeze(2)

        indices = torch.argmax(attention, 1).T.unsqueeze(2).expand(-1, -1, left_vectors.shape[2])
        right_difference = right_vectors - left_vectors.gather(0, indices)
        right_difference *= right_mask.T.unsqueeze(2)
        
        return left_difference, right_difference, left_mask, right_mask


class WeightedSentenceDifference(nn.Module):
    def __init__(self, we_dim, weights_dim=None, max_length=128):
        super(WeightedSentenceDifference, self).__init__()
        self.sd = SentenceDifference(max_length)
        if weights_dim is None:
            weights_dim = [400, 20]
        self.token_weight = sequential_neural_network([we_dim] + weights_dim, nn.ReLU, 0.05)
    def forward(self, left_vectors, left_lengths, right_vectors, right_lengths):
        left_difference, right_difference, left_mask, right_mask = self.sd(left_vectors, left_lengths, right_vectors, right_lengths)
        left_difference *= self.token_weight(left_vectors).max(2).values.unsqueeze(2)
        right_difference *= self.token_weight(right_vectors).max(2).values.unsqueeze(2)
        
        return left_difference, right_difference, left_mask, right_mask


class SentenceEncoder(nn.Module):
    def __init__(self, we_dim=300, hidden_dim=60, max_length=128):
        super(SentenceEncoder, self).__init__()
        from torch.nn.parameter import Parameter
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_length + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)
        self.proj = nn.Parameter(torch.ones((we_dim, hidden_dim)))
        self.dropout = nn.Dropout(p=0.05)
    def forward(self, sentence, size):
        max_sentence_length = sentence.shape[0]
        if max_sentence_length < 2:
            return sentence
        else:
            projected = torch.matmul(sentence, self.proj)
            projected = projected / torch.clamp((projected * projected).sum(2).unsqueeze(2), 1e-5)
            weight = torch.log(torch.zeros(list(projected.shape[:2]) + [3])).to(sentence.device)
            weight[1:, :, 0] = (projected[1:, :, :] * projected[:-1, :, :]).sum(2)
            weight[:, :, 1] = torch.ones(projected.shape[:2])
            weight[:-1, :, 2] = weight[1:, :, 0]
            mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_sentence_length + 1)]).T.bool()
            weight[:, :, 0][~mask_matrix[size].T] = weight[0, 0, 0] 
            weight[:, :, 2][~mask_matrix[torch.clamp(size - 1, 0)].T] = weight[0, 0, 0]
            weight = torch.softmax(weight, 2)
            
            output = weight[:, :, 1].unsqueeze(2) * sentence
            output[1:, :, :] += weight[1:, :, 0].unsqueeze(2) * sentence[:-1, :, :]
            output[:-1, :, :] += weight[:-1, :, 2].unsqueeze(2) * sentence[1:, :, :]
            
            return self.dropout(output)


class BilinearSentenceEncoder(nn.Module):
    def __init__(self, we_dim=300, max_length=128):
        super(BilinearSentenceEncoder, self).__init__()
        from torch.nn.parameter import Parameter
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_length + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)
        self.bilinear = nn.Bilinear(we_dim, we_dim, 1, bias=False)
        self.dim = we_dim
        self.dropout = nn.Dropout(p=0.05)
    def forward(self, sentence, size):
        max_sentence_length = sentence.shape[0]
        if max_sentence_length < 2:
            return sentence
        else:
            weight = torch.log(torch.zeros(list(sentence.shape[:2]) + [3])).to(sentence.device)
            weight[1:, :, 0] = 0.5 * (self.bilinear(sentence[1:, :, :], sentence[:-1, :, :]) + self.bilinear(sentence[:-1, :, :], sentence[1:, :, :])).squeeze(2)
            weight[:, :, 1] = self.bilinear(sentence, sentence).squeeze(2)
            weight[:-1, :, 2] = weight[1:, :, 0]
            mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_sentence_length + 1)]).T.bool()
            weight[:, :, 0][~mask_matrix[size].T] = weight[0, 0, 0]
            weight[:, :, 2][~mask_matrix[torch.clamp(size - 1, 0)].T] = weight[0, 0, 0]
            self.weight = weight
            weight = torch.softmax(weight / self.dim, 2)
            
            output = weight[:, :, 1].unsqueeze(2) * sentence
            output[1:, :, :] += weight[1:, :, 0].unsqueeze(2) * sentence[:-1, :, :]
            output[:-1, :, :] += weight[:-1, :, 2].unsqueeze(2) * sentence[1:, :, :]
            
            return self.dropout(output)


class SentenceDifferenceEncoding(nn.Module):
    def __init__(self, we_dim, sent_diff=None, hidden_dim=100, weights_dim=None):
        super(SentenceDifferenceEncoding, self).__init__()
        self.dim = we_dim
        self.conv1 = SentenceConv(1, hidden_dim, self.dim)
        if sent_diff is None:
            self.sd = WeightedSentenceDifference(self.dim)
        else:
            self.sd = sent_diff
        if weights_dim is None:
            weights_dim = [60]
        self.dense = sequential_neural_network([hidden_dim * 2] + weights_dim, nn.ReLU)
    def forward(self, l_lengths, l_vectors, r_lengths, r_vectors):
        features = []

        l_c, r_c, l_mask, r_mask = self.sd(l_vectors, l_lengths, r_vectors, r_lengths)
        for the_c, the_mask in [(l_c, l_mask), (r_c, r_mask)]:
            features.append(self.conv1(the_c.abs(), the_mask.T))
        return self.dense(torch.cat(features, 1))


class SentenceLSTMDifferenceEncoding(nn.Module):
    def __init__(self, we_dim, sent_diff=None, hidden_dim=100, weights_dim=None):
        super(SentenceLSTMDifferenceEncoding, self).__init__()
        self.dim = we_dim
        self.lstm  = nn.LSTM(self.dim, hidden_dim)
        if sent_diff is None:
            self.sd = WeightedSentenceDifference(self.dim)
        else:
            self.sd = sent_diff
        if weights_dim is None:
            weights_dim = [60]
        self.dense = sequential_neural_network([hidden_dim * 2] + weights_dim, nn.ReLU)
    def forward(self, l_lengths, l_vectors, r_lengths, r_vectors):
        features = []

        l_c, r_c, _, __ = self.sd(l_vectors, l_lengths, r_vectors, r_lengths)
        for the_c, the_size in [(l_c, l_lengths), (r_c, r_lengths)]:
            packed = pack_padded_sequence(the_c, lengths=the_size, batch_first=False, enforce_sorted=False)
            features.append(self.lstm(packed)[1][0][0])
        return self.dense(torch.cat(features, 1))

# class SequenceMask(nn.Module):
#     def __init__(self, max_len=128):
#         super(SequenceMask, self).__init__()
#         mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_len + 1)]).T
#         self.register_buffer('mask_matrix', mask_matrix)

#     def forward(self, x):
#         return self.mask_matrix[left_lengths, :left_vectors.shape[0]]
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)