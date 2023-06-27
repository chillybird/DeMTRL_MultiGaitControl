# -*- coding:utf-8 -*-
# @Time : 2022/2/27 14:05
# @Author: zhcode
# @File : main.py.py

from train import train_mappo
from utils.common import load_args_from_yaml

if __name__ == '__main__':
    args = load_args_from_yaml('train_args.yaml')
    # train agents in mpe environment
    common_args = {
        "multi_task": True,
        "use_eval": True,
        "eval_interval": 100,
        # Value Normalization
        "use_valuenorm": True,
        "use_popart": False,
        "seed": 0,
        "n_training_threads": 8,
        "n_rollout_threads": 100, # 24

        "max_length": 1000,
        "episode_length": 1000,
        "eval_episodes": 1,
        # "max_length": 1000,
        # "episode_length": 1000,
        "gamma": 0.996,

        "num_agent": 4,     # num of all agents
        # "use_wandb": True,
        "use_wandb": False,
        "user_name": "zhsample",
        "wandb_name": "zhsample",
        "hidden_size": 64,
        "num_mini_batch": 4,
        "ppo_epoch": 20,
        "use_shared_critic": False,
        "share_policy": False,
        "use_centralized_V": True,

        "eval_env": False,
        "save_interval": 100,
    }

    group_args = {
        "multi_gait_task_0": {
            "algorithm_name": "rmappo",
            "group": "mamc_fixed_start_multi_critic",
            "experiment_name": "laikago_gait_learning",

            "use_linear_decay_std": False,
            "use_linear_lr_decay": False,

            "use_adaptive_learning_rate": True,

            "lr": 0.00005,
            "critic_lr": 0.0005,
            # "fix_critic_lr": 5e-4,
            "layer_N": 2,

            "multi_task": True,
            "homo_agent": True,
            "update_statistics": True,

            "thresh": 0.01,
            "fixed_policy_lr": False,
            "episode_length": 1000,

            "gait_idx": 0,
            "gait_type_len": 3,
            "gait_idxes": [0, 1, 2],
            "env_config": "cfg",
            "seed": [1, 2, 3]
        },
        # other training instance ...
    }
    args.update(common_args)
    for group_name in group_args.keys():
        if isinstance(group_args[group_name]["seed"], list):
            for seed in group_args[group_name]["seed"]:
                print(f"run group '{group_name} seed {seed}'")
                group_args[group_name]['seed'] = seed
                args.update(group_args[group_name])
                train_mappo(args)
        else:
            print(f"run group '{group_name}'")
            args.update(group_args[group_name])
            train_mappo(args)
