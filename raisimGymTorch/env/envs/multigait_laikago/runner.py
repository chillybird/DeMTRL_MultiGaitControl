from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.multigait_laikago import RaisimGymEnv
from raisimGymTorch.env.bin.multigait_laikago import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import math
import time
import raisimGymTorch.algorithms.ppo.module as ppo_module
import raisimGymTorch.algorithms.ppo.ppo as PPO
import numpy as np
import torch
import datetime
import argparse

from mt_env import MultiTaskWrapper
from tensorboardX import SummaryWriter

# task specification
task_name = "laikago_locomotion_compare_original"
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
iter_num = 410

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
num_threads = cfg['environment']['num_threads']

# create multi-task env
env = MultiTaskWrapper(VecEnv(
    RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
    cfg['environment'],
    gait_type_len=3
))

env_num = env.env_num
gait_names = env.gait_names
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
print("obs_dim: ", ob_dim, " act_dim: ", act_dim)
print("env num:", env.num_envs)
print("n_steps: ", n_steps)

signal_scale = 0.3
use_eval = True


def eval_env(actor, critic, ppo, saver, episode, seed):
    print("Visualizing and evaluating the current policy")
    eval_args = {
        "env_num": 100,
        "noise": 0.05,
        "sampler": NormalSampler(act_dim),
        "iter_num": iter_num,
        "seed": seed,
        "device": 'cpu'
    }
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
    }, saver.data_dir + "/full_" + str(episode) + '.pt')
    eval_actor = ppo_module.Actor(cfg['architecture']['policy_net'], ob_dim, act_dim, **eval_args)
    eval_actor.load_state_dict(torch.load(saver.data_dir + "/full_" + str(episode) + '.pt')['actor_state_dict'])

    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("W%Y-%m-%d-%H-%M-%S") + "policy_" + str(episode) + '.mp4')

    for step in range(n_steps):     # n_steps * 2
        # time.sleep(0.08)
        with torch.no_grad():
            frame_start = time.time()
            obs = env.observe(False)
            action_ll, _ = eval_actor.sample(torch.from_numpy(obs).cpu(), deterministic=True)
            reward_ll, dones = env.step(action_ll)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()

    env.save_scaling(saver.data_dir, str(episode))


def train(mode, weight_path, seed=0):
    time_label = datetime.datetime.now().strftime("%Y_%m_%d_[%H_%M_%S]")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", time_label)
    print(f"run dir:{run_dir}")
    tensorboard_writer = SummaryWriter(logdir=run_dir, comment="rl_multi_task")

    saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/" + task_name,
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
    avg_rewards = []
    actor_args = {
        "env_num": env.num_envs,
        "noise": 0.05,
        "sampler": NormalSampler(act_dim),
        "iter_num": iter_num,
        "seed": seed,
        "device": device
    }
    actor = ppo_module.Actor(cfg['architecture']['policy_net'], ob_dim, act_dim, **actor_args)
    critic = ppo_module.Critic(cfg['architecture']['value_net'], ob_dim, 1, device=device)
    # critic = ppo_module.MultiMLPCritic(cfg['architecture']['value_net'], ob_dim, 1, env.env_num, device=device)
    ppo = PPO.PPO(actor=actor,
                  critic=critic,
                  num_envs=cfg['environment']['num_envs'],
                  num_transitions_per_env=n_steps,
                  num_learning_epochs=4,
                  gamma=0.996,
                  lam=0.95,
                  num_mini_batches=4,
                  device=device,
                  log_dir=saver.data_dir,
                  shuffle_batch=False,
                  )
    if mode == 'retrain':
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

    env.reset(0)
    train_start_time = time.time()
    for update in range(iter_num):
        start = time.time()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.

        if use_eval and update % cfg['environment']['eval_every_n'] == 0:
            eval_env(actor, critic, ppo, saver, update, seed)

        env.reset(update)
        # actual training
        for step in range(n_steps): # episode_len
            obs = env.observe()
            action = ppo.act(obs)
            reward, dones = env.step(action)
            # print("reward: ", reward, "dones： ", dones)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)

        # take st step to get value obs
        obs = env.observe()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')

        # log train reward curve use tensorboard
        log_label = gait_names[env.env_choice]
        tensorboard_writer.add_scalar(f'ppo/{log_label}_average_reward', average_ll_performance, update)

    train_end_time = time.time()
    elapse_time = train_end_time - train_start_time
    print(f"total use : {int((elapse_time / 60) * 100) / 100} min {int((elapse_time % 60) * 100) / 100} s")



if __name__ == '__main__':
    # configuration
    laikago_parser = argparse.ArgumentParser()
    laikago_parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    laikago_parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
    args = laikago_parser.parse_args()
    mode = args.mode
    weight_path = args.weight

    seed_list = [1, 2, 3]
    for random_seed in seed_list:
        print(f"run experiment with random seed {random_seed}")
        train(mode, weight_path, random_seed)
