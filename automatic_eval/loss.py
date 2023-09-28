import torch


def length_regularizer(z):
    """
    z = binary tensor of shape N, T

    return: loss of shape N,
    """
    return torch.norm(z, dim=-1)


def continuity_regularizer(z):
    """
    z = binary tensor of shape N, T

    return: loss of shape N,
    """
    return (z[:, 1:] - z[:, :-1]).abs().sum(-1)
