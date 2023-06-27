import torch
import torch.nn as nn

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        active_func = [nn.Tanh, nn.ReLU][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        modules = [nn.Linear(input_dim, hidden_size), active_func()]

        for idx in range(layer_N - 1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(active_func())

        self.architecture = nn.Sequential(*modules)
        self.init_weights(self.architecture, init_method, gain)

    def forward(self, x):
        x = self.architecture(x)
        return x

    @staticmethod
    def init_weights(sequential, init_method, scale):
        [init_method(module.weight, gain=scale) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        tmp = x

        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)

        return x


if __name__ == '__main__':
    mlp = MLPLayer(10, 128, 2, True, True)
    print(mlp)
    for p in mlp.parameters():
        print(p)
