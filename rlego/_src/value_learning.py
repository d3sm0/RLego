__all__ = [
    "td_learning",
    "td_lambda",
    "sarsa",
    "sarsa_lambda",
    "expected_sarsa",
    "sarsa_lambda",
    "q_learning",
    "double_q_learning",
    "persistent_q_learning",
    "qv_learning",
    "qv_max",
    "q_lambda",
    "retrace",
    "categorical_l2_project",
    "categorical_td_learning",
    "categorical_double_q_learning",
    "categorical_q_learning",
    "quantile_regression_loss",
    "quantile_q_learning",
    "quantile_expected_sarsa"
]

from typing import Union

import torch
from rlego._src import multistep
from rlego._src import utils

Tensor = torch.Tensor


def td_learning(v_tm1: Tensor, r_t: Tensor, discount_t: Tensor, v_t: Tensor) -> Tensor:
    discounted_value = torch.einsum("s,s->s", discount_t, v_t)
    td = r_t + discounted_value - v_tm1
    return td


def td_lambda(v_tm1: Tensor, r_t: Tensor, discount_t: Tensor, v_t: Tensor, lambda_: float):
    target_tm1 = multistep.lambda_returns(r_t=r_t, discount_t=discount_t, v_t=v_t, lambda_=lambda_)
    return target_tm1 - v_tm1


def sarsa(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor, a_t: Tensor):
    target_tm1 = r_t + discount_t * utils.batched_index(q_t, a_t)
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def expected_sarsa(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor, probs_a_t: Tensor):
    target_tm1 = r_t + discount_t * torch.einsum("a, a", probs_a_t, q_t)
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def sarsa_lambda(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor, a_t: Tensor,
                 lambda_: float):
    v_t = utils.batched_index(q_t, a_t)
    target_tm1 = multistep.lambda_returns(r_t=r_t, discount_t=discount_t, v_t=v_t, lambda_=lambda_)
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def q_learning(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor) -> Tensor:
    q_t = q_t.max(-1).values
    td = r_t + discount_t * q_t - utils.batched_index(q_tm1, a_tm1)
    return td


def double_q_learning(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t_value: Tensor,
                      q_t_selector: Tensor) -> Tensor:
    target_tm1 = r_t + discount_t * utils.batched_index(q_t_value, q_t_selector.argmax(dim=-1))
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def persistent_q_learning(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor,
                          action_gap_scale: float):
    corrected_q_t = (1 - action_gap_scale) * q_t.max(dim=-1).values + action_gap_scale * utils.batched_index(q_t,
                                                                                                              a_tm1)
    target_tm1 = r_t + discount_t * corrected_q_t
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def qv_learning(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, v_t: Tensor):
    target_tm1 = r_t + discount_t * v_t
    return target_tm1 - utils.batched_index(q_tm1, a_tm1)


def qv_max(v_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor):
    target_tm1 = r_t + discount_t * q_t.max(dim=-1).values
    return target_tm1 - v_tm1


def q_lambda(q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor, q_t: Tensor, lambda_: float):
    v_tm1 = utils.batched_index(q_tm1, a_tm1)
    v_t = q_t.max(dim=-1).values
    target_tm1 = multistep.lambda_returns(r_t=r_t, discount_t=discount_t, v_t=v_t, lambda_=lambda_)
    return target_tm1 - v_tm1


def retrace(q_tm1: Tensor, q_t: Tensor, v_t: Tensor, r_t: Tensor, discount_t: Tensor, log_rho_t: Tensor,
            lambda_: Union[Tensor, float]) -> Tensor:
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
      log_rho_t: Log importance weight pi_target/pi_behavior evaluated at actions
        collected using behavior policy [1, ..., K - 1].
      lambda_: scalar or a vector of mixing parameter lambda.
    Returns:
      Retrace error.
    """

    c_t = log_rho_t.exp().clamp_max(1.) * lambda_

    # The generalized returns are independent of Q-values and cs at the final
    # state.
    target_tm1 = multistep.general_off_policy_returns_from_q_and_v(q_t, v_t, r_t, discount_t, c_t)
    return target_tm1 - q_tm1


def categorical_l2_project(z_p: Tensor, probs: Tensor, z_q: Tensor):
    kp = z_p.shape[0]
    kq = z_q.shape[0]

    # Construct helper arrays from z_q.
    d_pos = torch.roll(z_q, shifts=-1)
    d_neg = torch.roll(z_q, shifts=1)

    # Clip z_p to be in new support range (vmin, vmax).
    z_p = torch.clip(z_p, z_q[0], z_q[-1])[None, :]
    assert z_p.shape == (1, kp)

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[:, None]  # z_q[i+1] - z_q[i]
    d_neg = (z_q - d_neg)[:, None]  # z_q[i] - z_q[i-1]
    z_q = z_q[:, None]
    assert z_q.shape == (kq, 1)

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = torch.where(d_neg > 0, 1. / d_neg, torch.zeros_like(d_neg))
    d_pos = torch.where(d_pos > 0, 1. / d_pos, torch.zeros_like(d_pos))

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]
    d_sign = (delta_qp >= 0.).to(probs.dtype)
    assert delta_qp.shape == (kq, kp)
    assert d_sign.shape == (kq, kp)

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    probs = probs[None, :]
    assert delta_hat.shape == (kq, kp)
    assert probs.shape == (1, kp)

    return torch.sum(torch.clip(1. - delta_hat, 0., 1.) * probs, dim=-1)


def categorical_td_learning(v_atoms_tm1: Tensor, v_logits_tm1: Tensor, r_t: Tensor, discount_t: Tensor,
                            v_atoms_t: Tensor,
                            v_logits_t: Tensor):
    target_z = r_t + discount_t * v_atoms_t
    v_t_probs = torch.softmax(v_logits_t, dim=-1)
    target = categorical_l2_project(target_z, v_t_probs, v_atoms_tm1)
    return torch.nn.functional.cross_entropy(v_logits_tm1, target)


def categorical_q_learning(q_atoms_tm1: Tensor, q_logits_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor,
                           q_atoms_t: Tensor,
                           q_logits_t: Tensor):
    target_z = r_t + discount_t * q_atoms_t
    q_t_probs = torch.softmax(q_logits_t, dim=-1)
    q_t_mean = (q_t_probs * q_atoms_t.unsqueeze(0)).sum(dim=-1)
    pi_t = q_t_mean.argmax(dim=-1)
    p_target_z = torch.index_select(q_t_probs, dim=0, index=pi_t.unsqueeze(0)).squeeze(0)

    target = categorical_l2_project(target_z, p_target_z, q_atoms_tm1)
    return torch.nn.functional.cross_entropy(torch.index_select(q_logits_tm1, 0, a_tm1.unsqueeze(0)).squeeze(0), target)


def categorical_double_q_learning(q_atoms_tm1: Tensor, q_logits_tm1: Tensor, a_tm1: Tensor, r_t: Tensor,
                                  discount_t: Tensor,
                                  q_atoms_t: Tensor,
                                  q_logits_t: Tensor,
                                  q_t_selector: Tensor
                                  ):
    target_z = r_t + discount_t * q_atoms_t
    p_target_z = utils.batched_select(q_logits_t, q_t_selector.argmax(-1)).softmax(-1)
    target = categorical_l2_project(target_z, p_target_z, q_atoms_tm1)
    return torch.nn.functional.cross_entropy(utils.batched_select(q_logits_tm1, a_tm1), target)


def quantile_regression_loss(dist_src: Tensor, tau_src: Tensor, dist_target: Tensor, huber_param: float = 1.):
    delta = dist_target.unsqueeze(0) - dist_src.unsqueeze(1)
    weight = torch.abs(tau_src.unsqueeze(1) - (delta < 0.).to(torch.float32)).detach()
    if huber_param == 0:
        delta = torch.abs(delta)
    else:
        abs_delta = torch.abs(delta)
        quadratic = torch.clamp_max(abs_delta, max=huber_param)
        delta = 0.5 * quadratic.pow(2) + huber_param * (abs_delta - quadratic)
    return (weight * delta).sum(0).mean(-1)


def quantile_q_learning(dist_q_tm1: Tensor, tau_q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor,
                        dist_q_t_selctor: Tensor, dist_q_t: Tensor, huber_param: float = 0.):
    dist_qa_tm1 = torch.index_select(dist_q_tm1, 1, a_tm1.unsqueeze(0)).squeeze(1)
    a_t = torch.mean(dist_q_t_selctor, dim=0).argmax()
    dist_qa_t = torch.index_select(dist_q_t, 1, a_t.unsqueeze(0)).squeeze(1)

    dist_target = r_t + discount_t * dist_qa_t
    return quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target, huber_param)


def quantile_expected_sarsa(dist_q_tm1: Tensor, tau_q_tm1: Tensor, a_tm1: Tensor, r_t: Tensor, discount_t: Tensor,
                            dist_q_t: Tensor, probs_a_t: Tensor, huber_param: float = 0.):
    import functorch
    dist_qa_tm1 = torch.index_select(dist_q_tm1, 1, a_tm1.unsqueeze(0)).squeeze(1)
    dist_target = r_t + discount_t * dist_q_t

    per_action_qr = functorch.vmap(
        quantile_regression_loss, in_dims=(None, None, 1, None)
    )
    per_action_loss = per_action_qr(dist_qa_tm1, tau_q_tm1, dist_target, huber_param)
    return (per_action_loss * probs_a_t).sum(-1)
