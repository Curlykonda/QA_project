import torch
import torch.nn as nn

from src.modules.encoder import HighwayEncoder


class BiDAF_Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    adopted from: https://github.com/minggg/squad

    Word-level embeddings are further refined using a 2-layer Highway Encoder

    Args:
        word_vectors (torch.FloatTensor): matrix of shape n_words x embedding_dim,
                                            either pre-trained or randomly initialised embeddings
        d_hidden (int): Size of hidden activations.
        p_drop (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors: torch.FloatTensor, d_hidden, p_drop, n_layers=2, freeze_we=True):
        super(BiDAF_Embedding, self).__init__()

        self.dropout_module = nn.Dropout(p=p_drop)

        if word_vectors is not None:
            self.embed = nn.Embedding.from_pretrained(word_vectors, freeze=freeze_we)
        else:
            raise ValueError("Need to provide embedding matrix")

        self.proj = nn.Linear(word_vectors.size(1), d_hidden, bias=False)
        self.hwy = HighwayEncoder(n_layers=n_layers, d_hidden=d_hidden)

    def forward(self, x):
        """ x contain sequences encoded as word indices """
        emb = self.dropout_module(self.embed(x))   # (batch_size, seq_len, embed_size)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class Simple_Embedding(nn.Module):
    """ Embedding layer to map word indices to corresponding word embeddings """

    def __init__(self, word_vectors: torch.FloatTensor, p_drop=0., freeze_we=False):
        super(Simple_Embedding, self).__init__()

        self.dropout = nn.Dropout(p_drop)
        if word_vectors is not None:
            self.embed = nn.Embedding.from_pretrained(word_vectors, freeze=freeze_we)
        else:
            raise ValueError("Need to provide embedding matrix")

    def forward(self, x):
        """ x contain sequences encoded as word indices """
        return self.dropout(self.embed(x))
