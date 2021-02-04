import torch
import torch.nn as nn

from src.modules.embedding import *
from src.modules.encoder import *
from src.modules.attention import *

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        d_hidden (int): Number of features in the hidden state at each layer.
        p_drop (float): Dropout probability.
    """
    def __init__(self, word_vectors, d_hidden, p_drop=0.):
        super(BiDAF, self).__init__()
        self.emb = BiDAF_Embedding(word_vectors=word_vectors,
                                   d_hidden=d_hidden,
                                   p_drop=p_drop)
        # contextual emb, i.e. interaction of context words independent of query
        self.enc = BiLSTM_Encoder(d_input=d_hidden,
                                  d_hidden=d_hidden,
                                  n_layers=1,
                                  drop_prob=p_drop)
        # attention flow, i.e. query-context interactions
        self.att = BiDAFAttention(hidden_size=2 * d_hidden,
                                  drop_prob=p_drop)
        # modelling layer, interaction among context words conditioned on queries
        self.mod = BiLSTM_Encoder(d_input=8 * d_hidden,
                                  d_hidden=d_hidden,
                                  n_layers=2,
                                  drop_prob=p_drop)
        # output layer to produce start & end indices
        self.out = BiDAFOutput(d_hidden=d_hidden,
                               drop_prob=p_drop)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out