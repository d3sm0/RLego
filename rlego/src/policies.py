import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F


class BetaPolicy(torch.nn.Module):
    eps = 1e-4

    def __init__(self, input_features: int, action_dim: int):
        # Note that we can trasform any standard beta to a bounded beta in x in (a,b)
        # by x = z * (b-a) + a
        super(BetaPolicy, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_features, 2 * action_dim),
            nn.Unflatten(1, (action_dim, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch_dist.Distribution:
        alpha, beta = torch.split(self.linear(x), split_size_or_sections=1, dim=2)
        # we want alpha and beta > 0
        alpha = F.softplus(alpha)
        beta = F.softplus(beta)
        alpha = alpha * (1 - self.eps) + self.eps
        beta = beta * (1 - self.eps) + self.eps
        dist = torch_dist.Beta(concentration0=alpha, concentration1=beta)
        return dist


class GaussianPolicy(torch.nn.Module):
    def __init__(self, input_features: int, action_dim: int):
        super(GaussianPolicy, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_features, 2 * action_dim),
            nn.Unflatten(1, (action_dim, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch_dist.Distribution:
        mean, scale = torch.split(self.linear(x), split_size_or_sections=1, dim=2)
        dist = torch_dist.Normal(loc=mean.squeeze(), scale=F.softplus(scale.squeeze()))
        return dist


class SoftmaxPolicy(torch.nn.Module):
    def __init__(self, input_features: int, n_actions: int, tau: float = 1.):
        super(SoftmaxPolicy, self).__init__()
        self.tau = tau
        self.linear = nn.Linear(input_features, n_actions)

    def forward(self, x: torch.Tensor) -> torch_dist.Distribution:
        logits = self.linear(x)
        return torch_dist.Categorical(logits=1 / self.tau * logits)
