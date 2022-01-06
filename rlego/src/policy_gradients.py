import abc

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


def advantage_loss_fn(model: ActorCriticType, obs, actions, rewards, next_obs, dones, discount, model_prime: ActorCriticType = None):
    assert len(obs.shape) == 2
    assert len(rewards.shape) == 1
    assert len(next_obs.shape) == 2
    assert len(dones.shape) == 1

    assert model.action_shape is None or len(actions.shape) == len(model.action_shape) + 1
    assert model.observation_shape is None or len(obs.shape) == len(model.observation_shape) + 1

    if model_prime is None:
        model_prime = model
    ALL = torch.arange(obs.shape[0])

    with torch.no_grad():
        not_done = torch.logical_not(dones)
        value_target = target_value_fn(discount, model_prime, next_obs, not_done, rewards)

    next_values = model.critic(next_obs)
    q_values = rewards + discount * torch.einsum("s,s->s", not_done, next_values)
    advantage = q_values - value_target
    return advantage, advantage ** 2


def target_value_fn(discount, model_prime: ActorCriticType, next_obs, not_done, rewards):
    assert len(next_obs.shape) == 2
    assert len(rewards.shape) == 1

    next_value = model_prime.critic(next_obs)
    target_value = rewards + discount * torch.einsum("s,s->s", not_done, next_value)
    return target_value
