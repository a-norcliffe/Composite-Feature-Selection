"""Thresholding functions for CompFS."""

# third party
import torch


def make_lambda_threshold(lmd):
    # If the value is above a certain value l (lambda) return 1, otherwise 0.
    lmd = float(lmd)

    def l_func(p):
        return p >= torch.full_like(p, lmd)

    return l_func


def make_std_threshold(nsigma):
    # Choose which features are relevant in p relative to other features,
    # if value of feature is above mean + n standard deviations.
    nsigma = float(nsigma)

    def std_dev_func(p):
        mean = torch.mean(p)
        std = torch.std(p)
        return p >= torch.full_like(p, (mean + nsigma * std).item())

    return std_dev_func


def make_top_k_threshold(k):
    # Choose top k features.
    k = int(k)

    def top_k(p):
        ids = torch.topk(p, k)[1]
        out = torch.zeros_like(p)
        out[ids] = 1.0
        return out.int()

    return top_k
