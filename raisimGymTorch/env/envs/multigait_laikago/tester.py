from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.multigait_laikago import RaisimGymEnv
from raisimGymTorch.env.bin.multigait_laikago import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algorithms.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse

from mt_env import MultiTaskWrapper

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
env = MultiTaskWrapper(VecEnv(
    RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
    cfg['environment'],
    gait_type_len=3
))

# shortcuts
env_num = env.env_num
ob_dim = env.num_obs
act_dim = env.num_acts
gait_names = env.gait_names

gait_idx = 2

weight_path = args.weight
print("weight_path: ", weight_path)
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset(gait_idx)
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)

    eval_args = {
        "env_num": 100,
        "noise": 0.05,
        "sampler": NormalSampler(act_dim),
        "iter_num": 1,
        "seed": cfg['seed'],
        "device": 'cpu'
    }
    eval_actor = ppo_module.Actor(cfg['architecture']['policy_net'], ob_dim, act_dim, **eval_args)
    eval_actor.load_state_dict(torch.load(weight_path)['actor_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    max_steps = 1000
    for step in range(max_steps):
        time.sleep(0.01)
        with torch.no_grad():
            obs = env.observe(False)
            action_ll, _ = eval_actor.sample(torch.from_numpy(obs).cpu(), deterministic=True)
            reward_ll, dones = env.step(action_ll)
            reward_ll_sum = reward_ll_sum + reward_ll[0]

        contacts = env.contact_logging()
        print(" ".join([str(int(x)) for x in contacts[0]]), file=contact_logger)

        pos = env.position_logging()
        print(" ".join([str(x) for x in pos[0]]), file=position_logger)

        signal = env.get_phase()
        print(" ".join([str(x) for x in signal[0]]), file=signal_logger)

        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0