from typing import Tuple

import torch

T = torch.Tensor


def general_off_policy_returns_from_q_and_v(
        q_t: T,
        v_t: T,
        r_t: T,
        discount_t: T,
        c_t: T,
) -> T:
    """Calculates targets for various off-policy evaluation algorithms.
    Given a window of experience of length `K+1`, generated by a behaviour policy
    μ, for each time-step `t` we can estimate the return `G_t` from that step
    onwards, under some target policy π, using the rewards in the trajectory, the
    values under π of states and actions selected by μ, according to equation:
      Gₜ = rₜ₊₁ + γₜ₊₁ * (vₜ₊₁ - cₜ₊₁ * q(aₜ₊₁) + cₜ₊₁* Gₜ₊₁),
    where, depending on the choice of `c_t`, the algorithm implements:
      Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
      Harutyunyan's et al. Q(lambda)  c_t = λ,
      Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
      Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).
    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).
    Args:
      q_t: Q-values under π of actions executed by μ at times [1, ..., K - 1].
      v_t: Values under π at times [1, ..., K].
      r_t: rewards at times [1, ..., K].
      discount_t: discounts at times [1, ..., K].
      c_t: weights at times [1, ..., K - 1].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Off-policy estimates of the generalized returns from states visited at times
      [0, ..., K - 1].
    """

    # Work backwards to compute `G_K-1`, ..., `G_1`, `G_0`.
    g = r_t[-1] + discount_t[-1] * v_t[-1]  # G_K-1.
    returns = [g]
    for i in reversed(range(q_t.shape[0])):  # [K - 2, ..., 0]
        g = r_t[i] + discount_t[i] * (v_t[i] - c_t[i] * q_t[i] + c_t[i] * g)
        returns.insert(0, g)
    returns = torch.stack(returns)
    return returns


def lambda_returns(r_t: T,
                   discount_t: T,
                   v_t: T,
                   lambda_: float = 1.,
                   ) -> T:
    """Estimates a multistep truncated lambda return from a trajectory.
    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.
    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.
      rₜ₊₁ + γₜ₊₁ vₜ₊₁
      rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
      rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃
    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:
      Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:
      TD(λ):  `v_t` contains the state value estimates for each state under π.
      Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
      Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.
    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:
      Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
      V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)
    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).
    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).
    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).
    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).
    Args:
      r_t: sequence of rewards rₜ for timesteps t in [1, T].
      discount_t: sequence of discounts γₜ for timesteps t in [1, T].
      v_t: sequence of state values estimates under π for timesteps t in [1, T].
      lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Multistep lambda returns.
    """
    # assert_rank([r_t, discount_t, v_t, lambda_], [1, 1, 1, {0, 1}])
    # assert_type([r_t, discount_t, v_t, lambda_], float)
    # assert_equal_shape([r_t, discount_t, v_t])

    # If scalar make into vector.
    lambda_ = torch.ones_like(discount_t) * lambda_

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    returns = []
    g = v_t[-1]
    for i in reversed(range(v_t.shape[0])):
        g = r_t[i] + discount_t[i] * ((1 - lambda_[i]) * v_t[i] + lambda_[i] * g)
        returns.insert(0, g)

    returns = torch.stack(returns)
    return returns


def discounted_returns(r_t: T,
                       discount_t: T,
                       v_t: T) -> T:
    bootstrapped_v = torch.ones_like(discount_t) * v_t
    return lambda_returns(r_t, discount_t, bootstrapped_v, lambda_=1)


def pad_n_step(r_t, discount_t: T, v_t, n: int, lambda_t=1) -> Tuple[T, T, T, T, T, int]:
    assert v_t.ndim == 2  # B x T
    seq_len = v_t.shape[1]
    pad_size = min(n - 1, seq_len)
    padding = v_t[:, -1:].repeat(1, pad_size)
    targets = torch.cat([v_t[:, n - 1:], padding], dim=-1)

    # Maybe change scalar lambda to an array.
    lambda_t = torch.ones_like(discount_t) * lambda_t
    # Pad sequences. Shape is now (T + n - 1,).
    pad_zeros = torch.zeros((v_t.shape[0], n - 1)).to(v_t.device)
    pad_ones = torch.ones((v_t.shape[0], n - 1)).to(v_t.device)
    r_t = torch.cat([r_t, pad_zeros], dim=1)
    discount_t = torch.cat([discount_t, pad_ones], dim=1)
    lambda_t = torch.cat([lambda_t, pad_ones], dim=1)
    padding = v_t[:, -1:].repeat(1, n - 1)
    v_t = torch.cat([v_t, padding], dim=1)
    return r_t, discount_t, v_t, targets, lambda_t, seq_len


def n_step_bootstrapped_returns(r_t: T, discount_t: T, v_t: T, targets: T, lambda_t: T, seq_len: int, n: int) -> T:
    """Computes strided n-step bootstrapped return targets over a sequence.
  The returns are computed according to the below equation iterated `n` times:
     Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
  When lambda_t == 1. (default), this reduces to
     Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).
  Args:
    r_t: rewards at times [1, ..., T].
    discount_t: discounts at times [1, ..., T].
    v_t: state or state-action values to bootstrap from at time [1, ...., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    lambda_t: lambdas at times [1, ..., T]. Shape is [], or [T-1].
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
  Returns:
    estimated bootstrapped returns at times [0, ...., T-1]
  """
    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i:i + seq_len]
        discount_ = discount_t[i:i + seq_len]
        lambda_ = lambda_t[i:i + seq_len]
        v_ = v_t[i:i + seq_len]
        targets = r_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)

    return targets


def truncated_generalized_advantage_estimation(
        r_t: T,
        discount_t: T,
        lambda_: T,
        values: T
) -> T:
    """Computes truncated generalized advantage estimates for a sequence length k.
    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).
    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347
    Note: This paper uses a different notation than the RLax standard
    convention that follows Sutton & Barto. We use rₜ₊₁ to denote the reward
    received after acting in state sₜ, while the PPO paper uses rₜ.
    Args:
      r_t: Sequence of rewards at times [1, k]
      discount_t: Sequence of discounts at times [1, k]
      lambda_: Mixing parameter; a scalar or sequence of lambda_t at times [1, k]
      values: Sequence of values under π at times [0, k]
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Multistep truncated generalized advantage estimation at times [0, k-1].
    """
    lambda_ = torch.ones_like(discount_t) * lambda_  # If scalar, make into vector.
    delta_t = r_t + discount_t * values[1:] - values[:-1]

    # Iterate backwards to calculate advantages.
    advantage_t = [torch.tensor(0.).to(r_t.device)]
    for t in reversed(range(delta_t.shape[0])):
        advantage_t.insert(0, delta_t[t] + lambda_[t] * discount_t[t] * advantage_t[0])
    adv = torch.stack(advantage_t)[:-1]
    return adv


def importance_corrected_td_errors(
        r_t: T,
        discount_t: T,
        rho_tm1: T,
        lambda_: T,
        values: T
) -> T:
    """Computes the multistep td errors with per decision importance sampling.
    Given a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a multistep temporal difference error δₜ(ρ,λ),
    by combining rewards, discounts, and state values, according to a mixing
    parameter `λ` and importance sampling ratios ρₜ = π(aₜ|sₜ) / μ(aₜ|sₜ):
      td-errorₜ = ρₜ δₜ(ρ,λ)
      δₜ(ρ,λ) = δₜ + ρₜ₊₁ λₜ₊₁ γₜ₊₁ δₜ₊₁(ρ,λ),
    where δₜ = rₜ₊₁ + γₜ₊₁ vₜ₊₁ - vₜ is the one step, temporal difference error
    for the agent's state value estimates. This is equivalent to computing
    the λ-return with λₜ = ρₜ (e.g. using the `lambda_returns` function from
    above), and then computing errors as  td-errorₜ = ρₜ(Gₜ - vₜ).
    See "A new Q(λ) with interim forward view and Monte Carlo equivalence"
    by Sutton et al. (http://proceedings.mlr.press/v32/sutton14.html).
    Args:
      r_t: sequence of rewards rₜ for timesteps t in [1, T].
      discount_t: sequence of discounts γₜ for timesteps t in [1, T].
      rho_tm1: sequence of importance ratios for all timesteps t in [0, T-1].
      lambda_: mixing parameter; scalar or have per timestep values in [1, T].
      values: sequence of state values under π for all timesteps t in [0, T].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Off-policy estimates of the multistep td errors.
    """

    v_tm1 = values[:-1]  # Predictions to compute errors for.
    v_t = values[1:]  # Values for bootstrapping.
    pad_ = torch.tensor([1.]).to(v_t.device)
    rho_t = torch.cat((rho_tm1[1:], pad_))  # Unused dummy value.
    lambda_ = torch.ones_like(discount_t) * lambda_  # If scalar, make into vector.

    # Compute the one step temporal difference errors.
    one_step_delta = r_t + discount_t * v_t - v_tm1

    # Work backwards to compute `delta_{T-1}`, ..., `delta_0`.
    delta, errors = 0.0, []
    for i in reversed(range(one_step_delta.shape[0])):
        delta = one_step_delta[i] + discount_t[i] * rho_t[i] * lambda_[i] * delta
        errors.insert(0, delta)

    errors = torch.stack(errors)
    errors = rho_tm1 * errors
    return errors
