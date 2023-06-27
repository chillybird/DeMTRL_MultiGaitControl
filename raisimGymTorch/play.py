import os
import time

import numpy as np
import torch
import argparse
from utils.common import load_args_from_yaml
from algorithms.algorithm.r_actor_critic import R_Actor

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.multigait_laikago import RaisimGymEnv


def _t2n(x):
    return x.detach().cpu().numpy()


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

args = load_args_from_yaml('train_args.yaml')
args.update({
    "num_agents": 4,  # num of all agents
    "eval_steps": 1000,
    "use_valuenorm": True,
    "use_popart": False,
    # "seed": 1,
    "hidden_size": 64,
    "layer_N": 2,
    "delay_time": 0.01,
    "test_mode": False,
    # "test_mode": True,

    "multi_task": True,
    "homo_agent": True,
})

task_args = {
    "gait_transitions": [(0, 1)], # [(changed_step, changed_gait), ...]
    "gait_idxes": [0, 1, 2],
    "model_dir": "models/multi_gaits",
}

args.update(task_args)
eval_args = argparse.Namespace(**args)

# env init
task_path = os.path.dirname(os.path.realpath(__file__))
task_name = "multigait_laikago"
home_path = "/home/zhang/raisim_ws/raisimLib"
cfg = YAML().load(open(task_path + f"/env/envs/{task_name}/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1

eval_envs = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                   cfg['environment'], num_env=cfg['environment']['num_envs'], ma_env=True,
                   multi_task=eval_args.multi_task, homo_agent=eval_args.homo_agent, gait_type_len=len(eval_args.gait_idxes))

eval_args.env_name = eval_envs.gait_names


iteration_number = eval_args.model_dir.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]

eval_envs.load_scaling(eval_args.model_dir, int(iteration_number))

obs_space = eval_envs.observation_space[0]
act_space = eval_envs.action_space[0]

policy_idx = [i for i in range(eval_args.num_agents)]
group_num = 4
if eval_args.homo_agent:
    group_num = 2
    policy_idx = [0, 0, 1, 1]

actors = [R_Actor(eval_args, obs_space, act_space, device) for _ in range(group_num)]

for agent_id in range(group_num):
    policy_actor_state_dict = torch.load(str(eval_args.model_dir) + '/actor_agent' + str(agent_id) + '.pt',
                                         map_location=lambda storage, loc: storage)
    actors[agent_id].load_state_dict(policy_actor_state_dict)
    actors[agent_id].eval()

if __name__ == '__main__':
    eval_rnn_states = np.zeros((eval_envs.n_rollout_threads, eval_args.num_agents, eval_args.recurrent_N, eval_args.hidden_size),
                               dtype=np.float32)
    eval_masks = np.ones((eval_envs.n_rollout_threads, eval_args.num_agents, 1), dtype=np.float32)
    obs, share_obs = eval_envs.ma_reset(update_statistics=False)

    eval_reward = 0.0

    transition_steps = [trans[0] for trans in eval_args.gait_transitions]
    transition_gaits = [trans[1] for trans in eval_args.gait_transitions]
    prev_git_idx = 0
    gait_idx = 0

    eval_envs.turn_on_visualization()
    for step in range(eval_args.eval_steps):
        if gait_idx < len(eval_args.gait_transitions) and step == transition_steps[gait_idx]:
            eval_envs.change_gait(eval_args.gait_idxes[transition_gaits[gait_idx]], task_idx=transition_gaits[gait_idx])
            prev_git_idx = gait_idx
            gait_idx = gait_idx + 1

        if len(eval_args.gait_transitions) > 1 and 0 < step < (transition_steps[prev_git_idx] + 30) and step > (transition_steps[prev_git_idx] - 30):
            time.sleep(eval_args.delay_time * 4)
        time.sleep(eval_args.delay_time)
        eval_actions = []
        for agent_id in range(eval_args.num_agents):
            eval_action, _, eval_rnn_state = actors[policy_idx[agent_id]](obs[:, agent_id],
                                                              eval_rnn_states[:, agent_id],
                                                              eval_masks[:, agent_id],
                                                              deterministic=True)
            eval_action = eval_action[0].detach().cpu().numpy()
            eval_actions.append(eval_action)
            eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

        eval_actions = np.array(eval_actions, dtype=np.float32)

        if eval_args.test_mode:
            eval_actions = np.zeros_like(eval_actions)

        obs, share_obs, rewards, dones = eval_envs.ma_step(eval_actions, update_statistics=False, last_action=(step == eval_args.eval_steps - 1))

        eval_rnn_states[dones == True] = np.zeros(((dones == True).sum(), eval_args.recurrent_N, eval_args.hidden_size),
                                                  dtype=np.float32)
        eval_masks = np.ones((eval_envs.n_rollout_threads, eval_args.num_agents, 1), dtype=np.float32)
        eval_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        eval_reward = eval_reward + rewards[0][0]

    eval_envs.turn_off_visualization()
    print("Eval mean reward: ", eval_reward / eval_args.eval_steps)
