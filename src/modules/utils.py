import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    eps= 1e30
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -eps # mask value of 0 indicates that it should be masked off
    logits.masked_fill_((1-mask).bool(), -eps)
    #RuntimeError: Expected object of scalar type Bool but got scalar type Float for argument #2 'mask' in call to _th_masked_fill_bool_
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs



class EMA:
    """Exponential moving average of model parameters.

    TF: 'When training a model, it is often beneficial to maintain moving averages of the trained parameters.
    Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.'

    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                #self.shadow[name] -= (1.0 - decay) * (self.shadow[name] - param.data)

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]