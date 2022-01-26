from typing import NamedTuple

import torch

T = torch.Tensor


class VTraceOutput(NamedTuple):
    pg_advantage: T
    target_tm1: T
    q_estimate: T


def vtrace_target(v_tm1: T, v_t: T, r_t: T, discount_t: T, rho_tm1: T, lambda_: float = 1.,
                  clip_rho_threshold: float = 1.):
    c_tm1 = torch.clamp(rho_tm1, max=1.) * lambda_
    clipped_rhos_tm1 = torch.clamp(rho_tm1, max=clip_rho_threshold)

    td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

    err = 0.0
    errors = torch.zeros_like(v_t)
    for i in reversed(range(v_t.shape[0])):
        err = td_errors[i] + discount_t[i] * c_tm1[i] * err
        errors[i] = err
    target = errors + v_tm1
    return target


def vtrace_td_error_and_advantage(v_tm1: T, v_t: T, r_t: T, discount_t: T, rho_tm1: T, lambda_: float = 1.,
                                  clip_rho_threshold: float = 1.0,
                                  clip_pg_rho_threshold: float = 1.0):
    target_tm1 = vtrace_target(v_tm1, v_t, r_t, discount_t, rho_tm1, clip_rho_threshold=clip_rho_threshold,
                               lambda_=lambda_)
    v_t = torch.cat([lambda_ * target_tm1[1:] + (1 - lambda_) * v_tm1[1:], v_t[-1:]], dim=0)
    q_t = r_t + discount_t * v_t
    rho_clipped = torch.clamp(rho_tm1, max=clip_pg_rho_threshold)
    adv = rho_clipped * (q_t - v_tm1)
    return VTraceOutput(adv, target_tm1, q_t)
