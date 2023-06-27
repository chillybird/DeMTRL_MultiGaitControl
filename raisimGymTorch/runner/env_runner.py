# -*- coding:utf-8 -*-
# @Time : 2022/5/19 14:10
# @Author: zhcode
# @File : env_runner_new.py
import os
import time
import datetime
import wandb
import numpy as np
import torch
from runner.base_runner import Runner
from algorithms.algorithm.r_actor_critic import R_Actor


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        start = time.time()
        episodes = 400
        gait_num = self.all_args.gait_type_len
        gait_idxes = self.all_args.gait_idxes if self.all_args.multi_task else None
        gait_idx = self.all_args.gait_idx

        # record task average rewards
        task_avg_episode_rewards = np.zeros(gait_num)

        print(f"train gait idx {gait_idxes if self.all_args.multi_task else gait_idx}")
        if not self.all_args.multi_task:
            gait_num = 1
            self.envs.change_gait(gait_idx, task_idx=0)

        self.warmup()

        for episode in range(episodes + 1):
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(episode)
            if self.all_args.multi_task:
                self.task_idx = episode % gait_num
                # print("change gait ", self.task_idx)
                gait_idx = gait_idxes[self.task_idx]
                self.envs.change_gait(gait_idx, task_idx=self.task_idx)

            self.envs.ma_reset(update_statistics=self.update_statistics)
            log_rewards = np.zeros(self.n_rollout_threads)
            for step in range(self.episode_length):
                # Sample actions through input agents' observation to actor
                values, actions, action_log_probs, rnn_states, rnn_states_critic, action_means, action_stds = \
                    self.collect(step)

                # Observe reward and next obs
                obs, share_obs, rewards, dones = self.envs.ma_step(actions,
                                                                   update_statistics=self.update_statistics,
                                                                   last_action=(step == (self.episode_length - 1)))

                log_rewards = log_rewards + rewards[:, 0]
                rewards = rewards.reshape((*rewards.shape, 1)) 

                data = obs, share_obs, rewards, dones, values, actions, action_log_probs, rnn_states, \
                       rnn_states_critic, action_means, action_stds

                # insert data into buffer with step index
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            avg_log_reward = log_rewards.mean() / self.episode_length

            # compute task average episode reward
            task_avg_episode_rewards[self.task_idx] = task_avg_episode_rewards[self.task_idx] + 1 / (
                        (episode / gait_num) + 1) * (avg_log_reward - task_avg_episode_rewards[self.task_idx])

            print("Episode {}({}), reward {}, mean reward {}.".format(episode, self.env_name[gait_idx], avg_log_reward,
                                                                  task_avg_episode_rewards[self.task_idx]))

            if self.use_wandb:
                wandb.log({f'{self.env_name[gait_idx]}_avg_reward': avg_log_reward}, step=total_num_steps)
            else:
                self.writter.add_scalars(f'{self.env_name[gait_idx]}_avg_reward',
                                         {f'{self.env_name[gait_idx]}_avg_reward': avg_log_reward},
                                         total_num_steps)

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start)) if total_num_steps != 0 else 0))

                self.log_train(train_infos, total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.ma_reset(update_statistics=self.update_statistics)
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        action_mean_collector = []
        action_std_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            task_idx=self.task_idx)

            action_mean, action_std = self.trainer[agent_id].policy.get_mean_std()
            action_mean_collector.append(_t2n(action_mean))
            action_std_collector.append(_t2n(action_std))

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        # [self.envs, agents, dim]
        action_means = np.array(action_mean_collector).transpose((1, 0, 2))
        action_stds = np.array(action_std_collector).transpose((1, 0, 2))

        values = np.array(value_collector).transpose((1, 0, 2))
        actions = np.array(action_collector).transpose((1, 0, 2))
        action_log_probs = np.array(action_log_prob_collector).transpose((1, 0, 2))
        rnn_states = np.array(rnn_state_collector).transpose((1, 0, 2, 3))
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose((1, 0, 2, 3))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_means, action_stds

    def insert(self, data):
        obs, share_obs, rewards, dones, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, action_means, action_stds = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id], values[:, agent_id],
                                         rewards[:, agent_id], masks[:, agent_id],
                                         action_means[:, agent_id], action_stds[:, agent_id],
                                         None, active_masks[:, agent_id])

    @torch.no_grad()
    def eval(self, episode):
        # save model
        if not os.path.exists(str(self.save_dir) + f"/model_{episode}"):
            os.mkdir(str(self.save_dir) + f"/model_{episode}")

        policy_idx = [i for i in range(self.num_agents)]
        save_policy_idx = [i for i in range(self.num_agents)]
        if self.all_args.homo_agent:
            policy_idx = [0, 0, 1, 1]
            save_policy_idx = [0, 2]

        for idx in range(self.group_num):  # self.num_agents
            policy_actor = self.trainer[save_policy_idx[idx]].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/model_{episode}" + "/actor_agent" + str(idx) + ".pt")

        self.envs.save_scaling(str(self.save_dir) + f"/model_{episode}", str(episode))

        actors = [R_Actor(self.all_args, self.envs.observation_space[0], self.envs.action_space[0], self.device) for _ in range(self.group_num)]
        for agent_id in range(self.group_num):
            policy_actor_state_dict = torch.load(str(self.save_dir) + f"/model_{episode}" + "/actor_agent" + str(agent_id) + ".pt",
                                                 map_location=lambda storage, loc: storage)
            actors[agent_id].load_state_dict(policy_actor_state_dict)
            actors[agent_id].eval()

        print("eval start.")

        eval_steps = 1000
        gait_num = 1
        gait_idxes = []
        if self.all_args.multi_task:
            gait_num = self.all_args.gait_type_len
            gait_idxes = self.all_args.gait_idxes

        self.envs.turn_on_visualization()
        self.envs.start_video_recording(
            datetime.datetime.now().strftime("W%Y-%m-%d-%H-%M-%S") + "policy_" + str(episode) + '.mp4')

        for idx in range(gait_num):
            if self.all_args.multi_task:
                self.envs.change_gait(gait_idxes[idx], task_idx=idx)
            else:
                self.envs.change_gait(self.all_args.gait_idx, task_idx=idx)
            one_episode_rewards = []
            for eval_i in range(self.n_rollout_threads):
                one_episode_rewards.append([])

            eval_obs, eval_share_obs = self.envs.ma_reset(update_statistics=False)
            eval_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step_idx in range(eval_steps):
                frame_start = time.time()

                eval_actions_collector = []
                eval_rnn_states_collector = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, _, temp_rnn_state = actors[policy_idx[agent_id]](eval_obs[:, agent_id],
                                                                                     eval_rnn_states[:, agent_id],
                                                                                     eval_masks[:, agent_id],
                                                                                     deterministic=True)

                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose((1, 0, 2))
                # eval_actions = np.zeros_like(eval_actions)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones = self.envs.ma_step(eval_actions,
                                                                                       update_statistics=False,
                                                                                       last_action=(step_idx == eval_steps - 1))
                for eval_i in range(self.n_rollout_threads):
                    one_episode_rewards[eval_i].append(eval_rewards[eval_i, 0])

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

                frame_end = time.time()
                wait_time = 0.01 - (frame_end - frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

            print("Gait {} eval_average_episode_rewards is {}.".
                  format(gait_idxes[idx] + 1 if self.all_args.multi_task else 1, np.mean(np.sum(one_episode_rewards, axis=1) / eval_steps)))

        self.envs.stop_video_recording()
        self.envs.turn_off_visualization()

        print("eval end.")


