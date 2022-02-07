import abc

import torch
import torch.distributions as torch_dist

T = torch.Tensor


def vanilla_policy_gradient(log_prob: T, q_t: T) -> T:
    return log_prob * q_t.detach()


def policy_gradient(pi_t: torch_dist.Distribution, a_t: T, q_t: T) -> T:
    action_log_prob = pi_t.log_prob(a_t.detach())
    return vanilla_policy_gradient(action_log_prob, q_t)


def mdpo(pi_k: torch_dist.Distribution, pi_km1: torch_dist.Distribution, a_t: T, adv: T) -> T:
    rho_t = (pi_k.log_prob(a_t) - pi_km1.log_prob(a_t)).sum(dim=-1).exp()
    kl = torch.distributions.kl_divergence(pi_k, pi_km1).sum(dim=-1)
    return (rho_t * adv), kl


class ActorCriticType(torch.nn.Module, abc.ABC):
    action_shape = None
    observation_shape = None

    def critic(self, state: T) -> T:
        raise NotImplementedError

    def actor(self, state: T) -> T:
        raise NotImplementedError
