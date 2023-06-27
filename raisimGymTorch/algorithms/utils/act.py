import torch

from .distributions import DiagGaussian
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None, device=torch.device('cpu')):
        super(ACTLayer, self).__init__()
        action_dim = action_space.shape[0]
        self.dist = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args, device)

    def forward(self, x, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        actions = self.dist.sample(x, deterministic)
        action_log_probs, _ = self.dist.logprobs_and_entropy(x, actions)

        return actions, action_log_probs

    def evaluate_actions(self, x, action):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return action_log_probs, dist_entropy

    def get_mean_std(self):
        return self.dist.get_mean_std()


