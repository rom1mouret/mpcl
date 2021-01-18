from collections import defaultdict
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch
import numpy as np


class Triangle(nn.Module):
    def forward(self, x):
        return x.abs()


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def confidence(x: torch.Tensor):
    logp = torch.log_softmax(x, dim=1)
    return logp.max(dim=1, keepdim=True)[0]


def new_classifier(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim)
    )

def new_processor(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 8),
        Triangle(),
        nn.Linear(in_dim // 8, in_dim // 16),
        Triangle(),
        nn.Linear(in_dim // 16, out_dim, bias=False)
    )


def apply_transformations(images: torch.Tensor, transformations: list) -> torch.Tensor:
    # group images by transformation
    indices = defaultdict(list)
    for i, t in enumerate(transformations):
        indices[t].append(i)
    indices = list(indices.items())

    # apply the transformations
    batches = []
    for transf_type, idx in indices:
        batch = images[idx]
        if transf_type == 0:
            pass
        elif transf_type == 1:
            batch = TF.rotate(batch, 90)
        elif transf_type == 2:
            batch = TF.rotate(batch, 180)
        elif transf_type == 3:
            batch = TF.rotate(batch, 270)
        elif transf_type == 4:
            batch = TF.hflip(batch)
        elif transf_type == 5:
            batch = TF.vflip(batch)
        elif transf_type == 6:
            batch = -batch
        else:
            raise Exception("unknown transformation: %i" % t)
        batches.append(batch)
    result = torch.cat(batches, dim=0)

    # put images back in order
    order = []
    for _, idx in indices:
        order += idx
    reverse_ordering = np.empty(len(order))
    reverse_ordering[order] = np.arange(len(order))

    return result[reverse_ordering]
