import torch


def td_learning(current_value: torch.Tensor,
                r_t: torch.Tensor,
                discount_t: torch.Tensor,
                v_t: torch.Tensor,
                stop_grad: float = True) -> torch.Tensor:
    if stop_grad:
        v_t = v_t.detach()
    discounted_value = torch.einsum("s,s->s", discount_t, v_t)
    td = r_t + discounted_value - current_value
    return td


def q_learning(q_tm1: torch.Tensor,
               r_t: torch.Tensor,
               discount_t: torch.Tensor,
               q_t: torch.Tensor,
               stop_grad: float = True) -> torch.Tensor:
    q_t = q_t.max(-1).values
    if stop_grad:
        q_t = q_t.detach()
    td = r_t + discount_t * q_t - q_tm1
    return td
