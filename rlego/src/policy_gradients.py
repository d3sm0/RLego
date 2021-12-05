import torch.distributions as torch_dist


def vanilla_policy_gradient(log_prob, q_t):
    return -log_prob * q_t.detach()


def softmax_policy_gradient(logits, a_t, q_t):
    log_prob = torch_dist.Categorical(logits=logits).log_prob(a_t.detach())
    return vanilla_policy_gradient(log_prob, q_t)
