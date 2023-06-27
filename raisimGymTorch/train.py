# !/usr/bin/env python
import os
import time
import argparse
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.multigait_laikago import RaisimGymEnv


def train_mappo(args):
    global run
    all_args = argparse.Namespace(**args)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "models" / (
        all_args.env_name if isinstance(all_args.env_name, str) else "multi_task")

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name if isinstance(all_args.env_name, str) else "multi_task",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.group,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exist_run_nums = [int(str(folder.name).split('_')[0].split('run')[1]) for folder in run_dir.iterdir() if
                              str(folder.name).startswith('run')]
            if len(exist_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exist_run_nums) + 1)
        curr_run = curr_run + "_{}".format(time.strftime('%Y-%m-%d_%H-%M', time.localtime()))
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name if isinstance(all_args.env_name,
                                                                  str) else "multi_task") + "-" + str(
        all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    task_path = os.path.dirname(os.path.realpath(__file__))
    task_name = "multigait_laikago"
    home_path = "/home/zhang/raisim_ws/raisimLib"
    cfg = YAML().load(open(task_path + f"/env/envs/{task_name}/{all_args.env_config}.yaml", 'r'))
    envs = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                  cfg['environment'], num_env=cfg['environment']['num_envs'], ma_env=True,
                  multi_task=all_args.multi_task, homo_agent=all_args.homo_agent, gait_type_len=all_args.gait_type_len)
    num_agents = all_args.num_agent

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from runner.env_runner import EnvRunner as Runner
        print("Shared Policy.")
    else:
        from runner.env_runner import EnvRunner as Runner
        print("Separated Policy.")

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        # save summary log file to model dir.
        # runner.writter.export_scalars_to_json(str(runner.save_dir + '/summary.json'))
        runner.writter.close()
