from typing import Union

import torch
from rlego.src.multistep import general_off_policy_returns_from_q_and_v

T = torch.Tensor


def td_learning(v_tm1: T, r_t: T, discount_t: T, v_t: T) -> T:
    discounted_value = torch.einsum("s,s->s", discount_t, v_t)
    td = r_t + discounted_value - v_tm1
    return td


def q_learning(q_tm1: T, r_t: T, discount_t: T, q_t: T) -> T:
    q_t = q_t.max(-1).values
    td = r_t + discount_t * q_t - q_tm1
    return td


def retrace(q_tm1: T, q_t: T, v_t: T, r_t: T, discount_t: T, rho_t: T, lambda_: Union[T, float]) -> T:
    """Retrace .
    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).
    Args:
      q_tm1: Q-values at times [0, ..., K - 1].
      q_t: Q-values evaluated at actions collected using behavior
        policy at times [1, ..., K - 1].
      v_t: Value estimates of the target policy at times [1, ..., K].
      r_t: reward at times [1, ..., K].
      discount_t: discount at times [1, ..., K].
      rho_t: Log importance weight pi_target/pi_behavior evaluated at actions
        collected using behavior policy [1, ..., K - 1].
      lambda_: scalar or a vector of mixing parameter lambda.
    Returns:
      Retrace error.
    """

    c_t = rho_t.clamp_max(1.) * lambda_

    # The generalized returns are independent of Q-values and cs at the final
    # state.
    target_tm1 = general_off_policy_returns_from_q_and_v(q_t, v_t, r_t, discount_t, c_t)
    return target_tm1 - q_tm1
