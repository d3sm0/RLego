import abc

import rlego
import torch
import torch.distributions as torch_dist


def vanilla_policy_gradient(log_prob: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    return log_prob * q_t.detach()


def softmax_policy_gradient(logits: torch.Tensor, a_t: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    log_prob = torch_dist.Categorical(logits=logits).log_prob(a_t.detach())
    return vanilla_policy_gradient(log_prob, q_t)


def gaussian_policy_gradient(density_params, a_t: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    log_prob = torch_dist.Normal(*density_params).log_prob(a_t.detach()).sum(-1)
    return vanilla_policy_gradient(log_prob, q_t)


class ActorCriticType(torch.nn.Module, abc.ABC):
    action_shape = None
    observation_shape = None

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def actor(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


