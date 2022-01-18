import torch
from typing import Iterable


def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float = 1.,
) -> None:
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
