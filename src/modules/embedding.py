import torch
import torch.nn as nn
import torch.nn.functional as F


from src.modules.encoder import HighwayEncoder


class BiDAF_Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    adopted from: https://github.com/minggg/squad

    Word-level embeddings are further refined using a 2-layer Highway Encoder

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        d_hidden (int): Size of hidden activations.
        p_drop (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors : torch.Tensor, d_hidden, p_drop):
        super(BiDAF_Embedding, self).__init__()
        self.drop_prob = p_drop
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), d_hidden, bias=False)
        self.hwy = HighwayEncoder(n_layers=2, d_hidden=d_hidden)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb