import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

T = torch.Tensor


class BetaPolicy(torch.nn.Module):
    eps = 1e-3

    def __init__(self, input_features: int, action_dim: int):
        # Note that we can trasform any standard beta to a bounded beta in x in (a,b)
        # by x = z * (b-a) + a
        super(BetaPolicy, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_features, 2 * action_dim),
            nn.Unflatten(-1, (action_dim, 2)),
        )

    def forward(self, x: T) -> torch_dist.Distribution:
        policy_params = self.linear(x)
        alpha, beta = torch.split(policy_params, split_size_or_sections=1, dim=-1)
        # we want alpha and beta > 0
        alpha = F.softplus(alpha.squeeze(dim=-1))
        beta = F.softplus(beta.squeeze(dim=-1))
        alpha = alpha + self.eps
        beta = beta + self.eps
        dist = torch_dist.Beta(concentration0=alpha, concentration1=beta)
        return dist


class GaussianPolicy(torch.nn.Module):
    eps = 1e-3

    def __init__(self, input_features: int, action_dim: int):
        super(GaussianPolicy, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_features, 2 * action_dim),
            nn.Unflatten(-1, (action_dim, 2)),
        )

    def forward(self, x: T) -> torch_dist.Distribution:
        policy_params = self.linear(x)
        mean, scale = torch.split(policy_params, split_size_or_sections=1, dim=-1)
        dist = torch_dist.Normal(loc=mean.squeeze(-1), scale=F.softplus(scale.squeeze(-1)) + self.eps)
        return dist


class SoftmaxPolicy(torch.nn.Module):
    def __init__(self, input_features: int, n_actions: int, tau: float = 1.):
        super(SoftmaxPolicy, self).__init__()
        self.tau = 1 / tau
        self.linear = nn.Linear(input_features, n_actions, bias=False)

    def forward(self, x: T) -> torch_dist.Distribution:
        logits = self.linear(x)
        return torch_dist.Categorical(logits=self.tau * logits)
