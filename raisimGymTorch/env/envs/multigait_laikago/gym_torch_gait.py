import os
import time
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.multigait_laikago import RaisimGymEnv
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv

import numpy as np

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config`
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 2

env = VecEnv(RaisimGymEnv(home_path + "/rsc",
             dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # 26 (w/ HAA joints fixed)
act_dim = env.num_acts

print("obs dim:", ob_dim, "act dim: ", act_dim)

num_steps = 1000

env.turn_on_visualization()
for k in range(3):
    env.change_gait(k+1, 0.2)
    ma_obs, ma_share_obs = env.ma_reset(False)

    # print(ma_obs[0])
    # print(ma_share_obs[0])

    for step in range(num_steps):

        time.sleep(0.01)
        actions = np.zeros(8)
        actions = np.array([actions for i in range(env.num_envs)]).astype(np.float32) # laikago 8, anymal 12

        actions = actions.reshape((*actions.shape[:-1], 4, 2))

        ma_obs, ma_share_obs, ma_reward, ma_done = env.ma_step(actions, update_statistics=False)
        # print("ma_obs: ", env.ma_step(actions, update_statistics=False))

        # print(ma_obs[0])
        # print(ma_share_obs[0])
        # print(ma_reward[0])
        # print(ma_done[0])

    input("p")

env.turn_off_visualization()

