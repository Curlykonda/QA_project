import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.utils import masked_softmax

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    adopted from: https://github.com/minggg/squad

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()

        self.drop_layer = nn.Dropout(p=drop_prob)

        self.c_lin_project = nn.Parameter(torch.zeros(hidden_size, 1)) # requires_grad=True (default)
        self.q_lin_project = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_lin_project = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_lin_project, self.q_lin_project, self.cq_lin_project):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        c2q = torch.bmm(s1, q) # compute context2query attention
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        q2c = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c) # query2context attention

        x = torch.cat([c, c2q, c * c2q, c * q2c], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        # apply dropout to context and question
        c = self.drop_layer(c)  # (bs, c_len, hid_size)
        q = self.drop_layer(q)  # (bs, q_len, hid_size)

        # s = alpha(c,q) = w_s^T [c;q;c*q]
        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_lin_project).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_lin_project).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_lin_project, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s