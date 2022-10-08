__all__ = ("VTraceOutput", "vtrace", "vtrace_td_error_and_advantage")

from typing import NamedTuple

import torch

from rlego._src.multistep import importance_corrected_td_errors

Tensor = torch.Tensor


class VTraceOutput(NamedTuple):
    pg_advantage: Tensor
    errors: Tensor
    q_estimate: Tensor

# TODO: return target  instead of errors
def vtrace(v_tm1: Tensor, v_t: Tensor, r_t: Tensor, discount_t: Tensor, rho_tm1: Tensor, lambda_: float = 1.,
           clip_rho_threshold: float = 1.):
    c_tm1 = torch.clamp_max(rho_tm1, clip_rho_threshold)
    errors = importance_corrected_td_errors(r_t, discount_t, c_tm1, torch.cat([v_tm1, v_t[-1:]], dim=0), lambda_=lambda_)
    return errors + v_tm1.detach() - v_tm1


def vtrace_td_error_and_advantage(v_tm1: Tensor, v_t: Tensor, r_t: Tensor, discount_t: Tensor, rho_tm1: Tensor,
                                  lambda_: float = 1.,
                                  clip_rho_threshold: float = 1.0,
                                  clip_pg_rho_threshold: float = 1.0):
    err = vtrace(v_tm1, v_t, r_t, discount_t, rho_tm1, clip_rho_threshold=clip_rho_threshold,
                    lambda_=lambda_)
    with torch.no_grad():
        target_tm1  = err + v_tm1
        q_bootstrap = torch.cat([lambda_ * target_tm1[1:] + (1 - lambda_) * v_tm1[1:], v_t[-1:]], dim=0)
        q_estimate = r_t + discount_t * q_bootstrap
        adv = torch.clamp_max(rho_tm1, clip_pg_rho_threshold) * (q_estimate - v_tm1)
    return VTraceOutput(adv, err, q_estimate)
