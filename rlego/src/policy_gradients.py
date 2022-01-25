import abc

import torch
import torch.distributions as torch_dist


def vanilla_policy_gradient(log_prob: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    return log_prob * q_t.detach()


def policy_gradient(dist: torch_dist.Distribution, a_t: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    action_log_prob = dist.log_prob(a_t.detach()).sum(-1)
    return vanilla_policy_gradient(action_log_prob, q_t)


class ActorCriticType(torch.nn.Module, abc.ABC):
    action_shape = None
    observation_shape = None

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def actor(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
