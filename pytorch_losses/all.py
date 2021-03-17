from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torchvision import models, transforms


# import numpy as np


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def cosine_loss(a, v, y):
    d = F.cosine_similarity(a, v)
    loss = F.binary_cross_entropy(d.unsqueeze(1), y)

    return loss
