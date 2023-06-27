import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, hidden_size, ob_dim, action_dim, env_num=1, noise=0.05, sampler=None, iter_num=1, seed=0,
                 device='cpu'):
        super(Actor, self).__init__()

        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.activate_fn = nn.ReLU

        self.architecture = MLP(hidden_size, self.activate_fn, ob_dim)
        self.distribution = MultivariateGaussianDiagonalCovariance(hidden_size[-1], action_dim, env_num, noise,
                                                                   sampler, iter_num=iter_num, seed=seed)
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs, deterministic=False):
        action_features = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(action_features, deterministic=deterministic)
        return actions, log_prob

    def evaluate(self, obs, actions):
        action_features = self.architecture.architecture(obs)
        return self.distribution.evaluate(action_features, actions)

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return [self.ob_dim]

    @property
    def action_shape(self):
        return [self.action_dim]


class Critic(nn.Module):
    def __init__(self, hidden_size, ob_dim, output_dim, device='cpu'):
        super(Critic, self).__init__()

        self.ob_dim = ob_dim
        self.activate_fn = nn.ReLU
        self.architecture = MLP(hidden_size, self.activate_fn, ob_dim)
        self.v_out = self.init(nn.Linear(hidden_size[-1], output_dim))
        self.architecture.to(device)
        self.v_out.to(device)

    def predict(self, obs):
        critic_features = self.architecture.architecture(obs)
        val = self.v_out(critic_features).detach()
        return val

    def evaluate(self, obs):
        critic_features = self.architecture.architecture(obs)
        val = self.v_out(critic_features)
        return val

    @property
    def obs_shape(self):
        return [self.ob_dim]

    def init(self, module):
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('relu')
        init_method(module.weight.data, gain=gain)
        return module


class MultiMLPCritic(nn.Module):
    """ multi network critic module """
    def __init__(self, hidden_size, ob_dim, output_dim, task_num, device='cpu'):
        super().__init__()

        self.ob_dim = ob_dim
        self.task_num = task_num
        self.activate_fn = nn.ReLU

        self.critics = nn.ModuleList()
        [self.critics.append(Critic(
            hidden_size=hidden_size,
            ob_dim=ob_dim,
            output_dim=output_dim
        )) for _ in range(task_num)]

        self.critics.to(device)

    def predict(self, obs):
        vals = torch.cat([critic.predict(obs) for critic in self.critics], -1)
        env_code = self.get_env_idx(obs)
        vals = vals[..., env_code].unsqueeze(-1)
        return vals

    def evaluate(self, obs):
        vals = torch.cat([critic.evaluate(obs) for critic in self.critics], -1)
        env_code = self.get_env_idx(obs)
        vals = vals[..., env_code].unsqueeze(-1)
        return vals

    @property
    def obs_shape(self):
        return [self.ob_dim]

    def get_env_idx(self, obs):
        if obs is None:
            obs = []
        obs_shape = obs.shape
        tmp = obs
        for _ in range(len(obs_shape) - 1):
            tmp = tmp[0]
        tmp = tmp[:self.task_num].cpu().numpy()
        return np.where(tmp == 1.0)[0][0]


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn
        self.gain = nn.init.calculate_gain('relu')

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [shape[-1]]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, input_dim, output_dim, size, noise, fast_sampler, direct_mean=None,
                 direct_std=None, iter_num=1, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, output_dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)

        self.iter_count = 0
        self.iter_num = iter_num
        self.init_rate = 1
        self.end_rate = 0.2
        self.rate = 1

        self.init_method = nn.init.uniform_
        self.init_output_scale = 0.01
        self.noise = noise

        self.input_size = input_dim
        self.action_dim = output_dim
        self.mean = direct_mean
        self.std = direct_std

        self._build_params()
        self.episode_std = self.std.detach().cpu().numpy()

        self.action_mean = None
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def _build_params(self):
        if self.mean is None:
            self.mean = self.init(nn.Linear(self.input_size, self.action_dim), lambda x: nn.init.constant_(x, 0),
                                  scale=self.init_output_scale)
        if self.std is None:
            self.std = nn.Parameter(self.noise * torch.ones(self.action_dim))

    def update(self):
        self.iter_count = self.iter_count + 1
        lerp = float(self.iter_count) / self.iter_num
        lerp = np.clip(lerp, 0.0, 1.0)
        self.rate = self.lerp(self.init_rate, self.end_rate, lerp)
        self.episode_std = self.std.detach().cpu().numpy()

        print("episode_std: ", self.episode_std)

    def get_action(self, logits):
        action_logits = self.mean(logits)
        action_logits = torch.tanh(action_logits).detach().cpu().numpy()

        return action_logits

    def sample(self, logits, deterministic=False):
        action_logits = self.get_action(logits)
        self.action_mean = action_logits

        if not deterministic:
            self.fast_sampler.sample(action_logits, self.episode_std, self.samples, self.logprob)
            return self.samples.copy(), self.logprob.copy()
        else:
            return action_logits, None

    def evaluate(self, logits, outputs):
        action_logits = torch.tanh(self.mean(logits))
        self.action_mean = action_logits
        distribution = Normal(action_logits, self.std.reshape(self.action_dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def init(self, module, bias_init, scale=1.0):
        nn.init.uniform_(module.weight.data, - scale, scale)
        bias_init(module.bias.data)
        return module

    def lerp(self, x, y, t):
        return (1 - t) * x + t * y

