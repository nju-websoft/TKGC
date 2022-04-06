import torch as th
import torch.nn as nn

class NumericRegression(nn.Module):
    def __init__(self, embed_dim, num_att) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.att_embed = nn.Embedding(num_att, embed_dim + 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, ent, att):
        att = self.att_embed(att)
        return self.activation((ent * att[:, :self.embed_dim]).sum(1) + att[:, 1])


class EntityRegression(nn.Module):
    def __init__(self, embed_dim, num_att) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.att_embed = nn.Embedding(num_att, embed_dim * embed_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, ent, att):
        att = self.att_embed(att)
        return (att.reshape(-1, self.embed_dim, self.embed_dim) * ent.unsqueeze(-1)).sum(1)


def main():
    embed_dim = 5
    reg = NumericRegression(embed_dim, 7)
    ent_embed = nn.Embedding(13, embed_dim)
    ent = th.randint(0, 13, (17, ))
    att = th.randint(0, 7, (17, ))
    val = th.rand((17,))
    
    ent = ent_embed(ent)
    print(reg(ent, att, val))

    embed_dim = 5
    reg = EntityRegression(embed_dim, 7)
    ent_embed = nn.Embedding(13, embed_dim)
    ent = th.randint(0, 13, (17, ))
    att = th.randint(0, 7, (17, ))
    val = th.rand((17,))

    ent = ent_embed(ent)
    print(reg(ent, att, val))


if __name__ == '__main__':
    main()