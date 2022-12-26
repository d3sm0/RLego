__all__ = ("polyak_update", "batched_select", "batched_index")

from typing import Iterable

import torch

Tensor = torch.Tensor


def polyak_update(
        params: Iterable[torch.nn.Parameter],
        target_params: Iterable[torch.nn.Parameter],
        tau: float = 1.,
) -> None:
    # zip does not raise an exception if length of parameters does not match.
    with torch.no_grad():
        for param, target_param in zip(params, target_params, strict=True):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def batched_index(values: Tensor, indices: Tensor):
    return torch.gather(values, index=indices.unsqueeze(dim=-1), dim=-1).squeeze(dim=-1)


def batched_select(values: Tensor, indices: Tensor):
    return torch.index_select(values, dim=0, index=indices.unsqueeze(0)).squeeze(0)
