import torch
import torch.nn as nn

from src.modules.embedding import *
from src.modules.encoder import *
from src.modules.attention import *
from transformers import RobertaModel

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
    def __init__(self, args, word_vectors, char_vectors, p_drop=0.):
        super(BiDAF, self).__init__()

        self.d_hidden = args.d_hidden
        # 0. determine if char emb will be used an adjust 'd_hidden' accordingly
        self.d_hidden_emb = self.d_hidden if char_vectors is None else 2 * self.d_hidden

        self.emb = BiDAF_Embedding(args,
                                   word_vectors=word_vectors,
                                   char_vectors=char_vectors,
                                   d_hidden=self.d_hidden_emb)
        # contextual emb, i.e. interaction of context words independent of query
        self.enc = BiLSTM_Encoder(d_input=self.d_hidden_emb,
                                  d_hidden=self.d_hidden,
                                  n_layers=1,
                                  drop_prob=p_drop)
        # attention flow, i.e. query-context interactions
        self.attn_flow_layer = BiDAFAttention(hidden_size=2 * self.d_hidden,
                                              drop_prob=p_drop)
        # modelling layer, interaction among context words conditioned on queries
        self.modelling_layer = BiLSTM_Encoder(d_input=8 * self.d_hidden,
                                              d_hidden=self.d_hidden,
                                              n_layers=2,
                                              drop_prob=p_drop)
        # output layer to produce logits of start & end indices
        self.out = BiDAFOutput(d_hidden=self.d_hidden,
                               drop_prob=p_drop)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs # create mask to not apply attention to padding tokens
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.attn_flow_layer(c_enc, q_enc,
                                   c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.modelling_layer(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class RobertaQA(nn.Module):
    """Roberta Encoder plus FC-layer as head for SQuAD.

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:

        freeze_enc (bool): Fix weights of pretrained BERT encoder
        freeze_emb (bool): Fix pretrained word embeddings of BERT encoder
        p_drop (float): Dropout probability.
    """
    def __init__(self, freeze_enc=True, freeze_emb=False, d_hidden=768, p_drop=0.1, n_outputs=2):
        super(RobertaQA, self).__init__()

        self.roberta_enc = RobertaModel.from_pretrained('roberta-base')
        self.freeze_enc = freeze_enc
        self.freeze_emb = freeze_emb

        # fix pre-trained Roberta parameters - only train QA-head & embeddings
        if freeze_enc:
            for param in self.roberta_enc.encoder.parameters():
                param.requires_grad = False

        if freeze_emb:
            for param in self.roberta_enc.embeddings.parameters():
                param.requires_grad = False

        self.qa_head = nn.Linear(d_hidden, n_outputs)
        assert n_outputs == 2 # output units for start and end index

        self.dropout = nn.Dropout(p_drop)
        #self.c_len = max_context_len + 1 # seq_len + [CLS]


    def forward(self, q_c_idxs, attn_mask):

        enc_out = self.roberta_enc(q_c_idxs, attention_mask=attn_mask)

        # (bs x seq_len x d_hidden)
        enc_seq = enc_out[0] # last hidden layer encodings for all tokens

        logits = self.qa_head(self.dropout(enc_seq)) # (bs, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # (bs x seq_len)
        end_logits = end_logits.squeeze(-1)

        # mask off padding positions
        eps = 1e30
        bool_mask = (1 - attn_mask).bool() # orig attn_mask is 0 for mask positions
        start_logits = start_logits.masked_fill_(bool_mask, -eps)
        # not sure if this retains information flow / gradient properly
        end_logits = end_logits.masked_fill_(bool_mask, -eps)

        return start_logits, end_logits


class QA_Net(nn.Module):

    def __init__(self):
        pass

    def forward(self, q_idxs, c_idxs, attn_mask):
        pass

