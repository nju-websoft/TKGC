import torch.nn as nn
import torch as th


class Modelassification(nn.Module):
    def __init__(self, dataset, n_hidden, node_dim, value_dim):
        self.att_embeds = nn.Embedding(len(dataset.attributes.i2w), node_dim)
        self.cate_embeds = nn.Embedding(len(dataset.categories.i2w), value_dim)

        self.classification['numerical'] = nn.Sequential([
            nn.Linear(n_hidden + node_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        ])

        self.classification['categorical'] = nn.Sequential([
            nn.Linear(n_hidden + node_dim + value_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, 1),
            nn.Sigmoid()
        ])
    
    def forward(self, input, mapping=None, heads=None, ent_embeds=None):
        output = dict()
        if 'numerical' in input:
            f, a, v, s, n = input['numerical']
            attribute = self.att_embeds(a)
            ent = ent_embeds(f)
            feature = th.cat([ent, attribute, v], 2)
            output['numerical'] = self.classification['numerical']
        elif 'categorical' in input:
            f, a, v, s, n = input['numerical']
            attribute = self.att_embeds(a)
            ent = ent_embeds(f)
            feature = th.cat([ent, attribute, v], 2)
            output['categorical'] = self.classification['categorical']
        return output


def main():
    import numpy as np
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    def TkgcDataset(base_dir):
        from pathlib import Path
        base_dir = Path(base_dir)
        train = TripleDataset(base_dir / 'train.csv')
        valid = TripleDataset(base_dir / 'validation.csv')
        test = TripleDataset(base_dir / 'test.csv')
        tokens = [c.get_tokens() for c in [train, valid, test]]
        tokens = np.unique(np.concatenate(tokens))
        print('Token set size:', tokens.shape[0])
        from vocabulary_table import VocabularyTable
        word_embedding_model = fasttext.load_model('wiki.en.bin')
        vocab = VocabularyTable(tokens, word_embedding_model)
        train.inject_vocab(vocab)
        valid.inject_vocab(vocab)
        test.inject_vocab(vocab)
        return train, valid, test, vocab
    # data
    train, valid, test, vocab = TkgcDataset('datasets/OKELE/' + task_name)

    train_loader = DataLoader(train, batch_size=256, num_workers=8)
    val_loader = DataLoader(valid, batch_size=256, num_workers=8)
    test_loader  = DataLoader(test, batch_size=256, num_workers=8)

    # model
    model = Modelassification(vocab.vectors, train.columns)

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=50, progress_bar_refresh_rate=310)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
    