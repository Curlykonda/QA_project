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
    def __init__(self, args, word_vectors: torch.FloatTensor, char_vectors, d_hidden, n_layers=2, freeze_we=True):
        super(BiDAF_Embedding, self).__init__()

        self.args = args
        self.p_drop = args.drop_prob
        self.d_hidden = d_hidden

        self.dropout_module = nn.Dropout(p=self.p_drop)

        # 1. Character Embedding Layer
        if char_vectors is not None:
            self.char_emb = nn.Embedding.from_pretrained(char_vectors)

            self.d_char_emb = args.char_dim  # input dimesion of char embs
            self.char_n_filters = args.char_n_filters
            self.char_kernel_size = args.char_kernel_size
            self.char_limit = args.char_limit if args.char_limit is not None else 16

            self.char_conv = nn.Sequential(
                nn.Conv2d(1, self.char_n_filters, (self.char_dim, self.char_kernel_size)),
                nn.ReLU(),
                nn.MaxPool1d(self.char_n_filters))

        else:
            self.char_emb = None
            self.char_conv = None
            self.d_char_emb = 0
            #self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
            # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        #

        if word_vectors is not None:
            self.word_emb = nn.Embedding.from_pretrained(word_vectors, freeze=freeze_we)
            self.d_word_emb = word_vectors.size(1)
        else:
            raise ValueError("Need to provide embedding matrix")

        self.proj = nn.Linear(self.d_word_emb + self.d_char_emb, self.d_hidden_eff, bias=False)
        self.hwy = HighwayEncoder(n_layers=n_layers, d_hidden=self.d_hidden_eff)

    def forward(self, x_w, x_c=None):
        """ x contain sequences encoded as word indices """

        # 2. word emb
        x_word = self.dropout_module(self.word_emb(x_w))   # (batch_size, seq_len, embed_size)

        # 1 char emb
        if self.char_emb is not None and x_c is not None:
            x_char = self.dropout_module(self.char_emb(x_c))   # (batch_size, seq_len, embed_size)
            emb = torch.cat([x_char, x_word], dim=2)
        else:
            emb = x_word

        # 3 highway
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
