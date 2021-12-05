from rlego.utils import _maybe_stop_grad
import torch


def td_learning(v_tm1: torch.Tensor,
                r_t: torch.Tensor,
                discount_t: torch.Tensor,
                v_t: torch.Tensor,
                stop_grad: float = True) -> torch.Tensor:
    _maybe_stop_grad(v_t, stop_grad)
    td = r_t + discount_t * v_t - v_tm1
    return td


def q_learning(q_tm1: torch.Tensor,
               r_t: torch.Tensor,
               discount_t: torch.Tensor,
               q_t: torch.Tensor,
               stop_grad: float = True) -> torch.Tensor:
    _maybe_stop_grad(q_t, stop_grad)
    td = r_t + discount_t * q_t.max(-1) - q_tm1
    return td
