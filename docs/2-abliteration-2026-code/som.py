"""som.py - multi-directional refusal via a Self-Organizing Map (Piras et al., AAAI 2026), compact re-implementation.

Train a 4x4 hexagonal SOM on harmful-prompt activations at one layer; each neuron minus the harmless
mean is one candidate direction. Settings follow the paper: T = 10,000 steps, lr0 = 0.01, sigma0 = 0.3,
both decaying as 1 / (1 + 2t/T), neurons initialised from random data rows (MiniSom's random_weights_init).

Two ways to remove k chosen directions:
  * ablate_sequential: P_k ... P_1 applied to every writer output (what the reference code does,
    github.com/pralab/som-refusal-directions @ d244c7d, one rank-1 weight edit per direction);
  * abliterate.ablate(model, V): projects out the whole span (orthonormalised with QR).
For non-orthogonal directions these differ: sequential projection keeps part of the span.
"""
import contextlib
import math

import torch

from abliterate import writer_modules


def train_som(X, nx=4, ny=4, T=10_000, lr0=0.01, sigma0=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = X.double()
    n = len(X)
    W = X[torch.randint(n, (nx * ny,), generator=g)].clone()
    order = torch.randint(n, (T,), generator=g)
    i, j = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    grid = torch.stack([i - 0.5 * (j % 2), j * math.sqrt(3) / 2], -1).reshape(-1, 2).double()
    D2 = torch.cdist(grid, grid) ** 2                       # squared distance on the hex lattice
    for t, k in enumerate(order.tolist()):
        x, decay = X[k], 1 + 2 * t / T
        bmu = ((W - x) ** 2).sum(1).argmin()
        h = torch.exp(-D2[bmu] / (2 * (sigma0 / decay) ** 2))
        W += (lr0 / decay) * h[:, None] * (x - W)
    counts = torch.bincount(torch.cdist(X, W).argmin(1), minlength=nx * ny)
    return W, counts


def som_directions(H_harmful, H_harmless, **kw):
    """Directions = neuron - harmless mean, sorted by how many harmful prompts each neuron wins."""
    W, counts = train_som(H_harmful, **kw)
    order = torch.argsort(counts, descending=True, stable=True)
    order = order[counts[order] > 0]
    return (W[order] - H_harmless.double().mean(0)).float(), counts[order]


def compose(x, V):
    """P_vk ... P_v1 x with unit rows v (list order)."""
    y = x.float()
    for v in V:
        y = y - (y @ v)[..., None] * v
    return y.to(x.dtype)


@contextlib.contextmanager
def ablate_sequential(model, V):
    V = torch.nn.functional.normalize(V.float(), dim=-1).to(model.device)
    hooks = [m.register_forward_hook(lambda mod, inp, out: compose(out, V)) for _, m in writer_modules(model)]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()
