import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.modules.utils import masked_softmax

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    adopted from: https://github.com/minggg/squad

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        n_layers (int): Number of layers in the highway encoder.
        d_hidden (int): Size of hidden activations.
    """
    def __init__(self, n_layers, d_hidden):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(d_hidden, d_hidden)
                                         for _ in range(n_layers)])
        self.gates = nn.ModuleList([nn.Linear(d_hidden, d_hidden)
                                    for _ in range(n_layers)])

        self.relu = nn.ReLU()
        #self.sigm = nn.Sigmoid()

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            #t = F.relu(transform(x))
            t = self.relu(transform(x))

            x = g * t + (1 - g) * x

        return x


class BiLSTM_Encoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        d_input (int): Size of a single timestep in the input.
        d_hidden (int): Size of the RNN hidden state.
        n_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, d_input, d_hidden, n_layers, drop_prob=0., bi=True):
        super(BiLSTM_Encoder, self).__init__()

        self.dropout_module = nn.Dropout(p=drop_prob)
        self.bi_lstm = nn.LSTM(d_input, d_hidden, n_layers,
                               batch_first=True,
                               bidirectional=bi,
                               dropout=drop_prob if n_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.bi_lstm.flatten_parameters()
        rnn_out, _ = self.bi_lstm(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        # both directions are concatenated implicitly
        rnn_out = rnn_out[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        return self.dropout_module(rnn_out)


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied to the model output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        d_hidden (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.

    :return
        log_p1 (float): softmax probability of start pointer of answer
        log_p2 (float): softmax probability of end pointer (to determine answer span)
    """
    def __init__(self, d_hidden, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * d_hidden, 1)
        self.mod_linear_1 = nn.Linear(2 * d_hidden, 1)

        # LSTM encoder for end indices applied on top of M (output of 2-layer
        self.rnn = BiLSTM_Encoder(d_input=2 * d_hidden,
                                  d_hidden=d_hidden,
                                  n_layers=1,
                                  drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * d_hidden, 1)
        self.mod_linear_2 = nn.Linear(2 * d_hidden, 1)

    def forward(self, att, mod, mask):
        """
        :param att: interaction between context and query captured in attention scores (shape: (batch_size, c_len, 8 * hidden_size))
        :param mod: context representations conditioned on query (shape: (batch_size, c_len, 2*hidden_size))
        :param mask: mask off padding tokens
        :return:
        """

        # Shapes: (batch_size, context_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2