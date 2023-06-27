# -*- coding:utf-8 -*-
# @Time : 2022/5/19 14:53
# @Author: zhcode
# @File : distributions_new.py

import numpy as np
import torch
import torch.nn as nn


# https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/net.py
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None,  device=torch.device('cpu')):
        super(DiagGaussian, self).__init__()
        self.args = args
        self.num_outputs = num_outputs
        self.device = device

        self.init_output_scale = 0.01
        self.noise = 0.05

        self.fc_mean = self.init(nn.Linear(num_inputs, num_outputs),
                                 lambda x: nn.init.constant_(x, 0), scale=self.init_output_scale)
        self.action_logstd = nn.Parameter(np.log(self.noise) * torch.ones((1, num_outputs)), requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        self.dist_mean = torch.zeros((1, num_outputs))

    def forward(self, x):
        x = torch.tanh(self.fc_mean(x))
        self.dist_mean = x

        return x, self.action_logstd.exp()

    def sample(self, x, deterministic):
        action_mean, action_std = self(x)

        if not deterministic:
            noise = torch.randn_like(action_mean)
            return action_mean + noise * action_std
        else:
            return action_mean

    def logprobs_and_entropy(self, x, actions):
        action_mean, action_std = self(x)

        delta = ((action_mean - actions) / action_std).pow(2) * 0.5
        log_prob = -(self.action_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = -(log_prob.exp() * log_prob).mean()  # policy entropy
        return log_prob.unsqueeze(-1), dist_entropy

    def init(self, module, bias_init, scale=1.0):
        nn.init.uniform_(module.weight.data, - scale, scale)
        bias_init(module.bias.data)
        return module

    def get_mean_std(self):
        action_shape = self.dist_mean.shape
        return self.dist_mean, self.action_logstd.exp()\
            .repeat(action_shape[0], *([1]*(len(action_shape) - 1)))

