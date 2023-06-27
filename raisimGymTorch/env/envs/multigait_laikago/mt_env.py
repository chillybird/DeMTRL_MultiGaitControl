import numpy as np


class MultiTaskWrapper(object):
    """ create a multi task env by wrapping raisim env api"""
    def __init__(self, env):
        assert env, "env can not be None"
        self._env = env
        self.env_choice = 0
        self.env_idxes = [0, 1, 2]
        self.env_num = len(self.env_idxes)
        self.num_obs = env.num_obs + self.env_num
        # print(env.num_obs, self.env_num)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, episode=None):
        assert episode is not None, "episode can not be None."
        self.env_choice = episode % self.env_num
        self._env.change_gait(self.env_idxes[self.env_choice], self.env_choice)
        print(f"change {self._env.gait_names[self.env_idxes[self.env_choice]]} gait")
        self._env.reset()

    def observe(self, update_statistics=True):
        # [envs, dims]
        obs = self._env.observe(update_statistics=update_statistics)
        obs_dim = obs.shape
        env_code = self._env.get_env_code()
        env_codes = np.tile(env_code, np.prod(obs_dim[:-1], axis=-1)).reshape((*obs.shape[:-1], self.env_num))

        return np.concatenate([env_codes, obs], axis=-1, dtype=np.float32)
