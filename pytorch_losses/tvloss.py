import torch
import torch.nn as nn
import torch.nn.functional as F

class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).
    The input must be 4D. If a target (second parameter) is passed in, it is
    ignored.
    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.
    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction="mean", eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError("p must be 1 or 2")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), "replicate")
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff ** 2 + y_diff ** 2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == "mean":
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        return diff
