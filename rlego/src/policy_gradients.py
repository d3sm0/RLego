import abc

import torch
import torch.distributions as torch_dist

T = torch.Tensor


def vanilla_policy_gradient(log_prob: T, q_t: T) -> T:
    return log_prob * q_t.detach()


def policy_gradient(dist: torch_dist.Distribution, a_t: T, q_t: T) -> T:
    action_log_prob = dist.log_prob(a_t.detach()).sum(-1)
    return vanilla_policy_gradient(action_log_prob, q_t)


class ActorCriticType(torch.nn.Module, abc.ABC):
    action_shape = None
    observation_shape = None

    def critic(self, state: T) -> T:
        raise NotImplementedError

    def actor(self, state: T) -> T:
        raise NotImplementedError
