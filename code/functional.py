import torch

def packed_sequence_mean(seq, embeddings, device='cuda'):
    avg = torch.zeros(seq.batch_sizes[0], embeddings.embedding_dim).to(device)
    cnt = torch.zeros(seq.batch_sizes[0]).to(device)
    vectors = embeddings(seq.data)
    
    offset = 0
    for bs in seq.batch_sizes:
        cnt[0:bs] += 1
        avg[0:bs] += vectors[offset:offset+bs]
        offset += bs

    return avg / cnt.view(-1, 1)