# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
from matplotlib import pyplot as plt
import numpy as np
import platform
import os
import gym.spaces as Spaces


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=5., num_env=1, ma_env=False, multi_task=False, homo_agent=None, gait_type_len = 4):  # 10.
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.gait_idx = 1  
        self.task_idx = 0

        self.gait_names = ["walk", "trot", "pace", "bound"]
        self.gait_type_len = gait_type_len

        self.n_rollout_threads = num_env

        self.num_agents = 4
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl

        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()

        self.observation_space = None
        self.share_observation_space = None
        self.action_space = None
        if ma_env:
            self.num_critic_obs = self.wrapper.getCriticObDim()  # 10
            self.num_actor_obs = self.wrapper.getActorObDim() + (
                self.gait_type_len if multi_task else 0)  # 8 + one-hot code

            self.share_observation_space = [
                Spaces.Box(-np.ones(self.num_critic_obs), np.ones(self.num_critic_obs), dtype=np.float32)
                for _ in range(self.num_agents)
            ]
            self.observation_space = [
                Spaces.Box(-np.ones(self.num_actor_obs), np.ones(self.num_actor_obs), dtype=np.float32)
                for _ in range(self.num_agents)
            ]
            ma_num_acts = int(self.num_acts / self.num_agents)
            self.action_space = [
                Spaces.Box(-np.ones(ma_num_acts), np.ones(ma_num_acts), dtype=np.float32)
                for _ in range(self.num_agents)
            ]

            # self.ma_actions = np.zeros([self.num_envs, self.num_agents, self.num_acts], dtype=np.float32)
            self.ma_actor_obs = np.zeros([self.num_envs, self.num_agents, self.num_actor_obs], dtype=np.float32)
            self.ma_critic_obs = np.zeros([self.num_envs, self.num_agents, self.num_critic_obs], dtype=np.float32)

        else:
            self.observation_space = Spaces.Box(-np.ones(self.num_obs), np.ones(self.num_obs), dtype=np.float32)
            self.action_space = Spaces.Box(-np.ones(self.num_acts), np.ones(self.num_acts), dtype=np.float32)

        print(f"train gait type len {gait_type_len}.")
        print("share observation space:", self.share_observation_space, "\nobservation space: ", self.observation_space, "\naction space: ", self.action_space)

        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._phase = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)

        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._CPG_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.reward_log = np.zeros([self.num_envs, cfg["n_reward"]], dtype=np.float32)
        self.position_log = np.zeros([self.num_envs, 3], dtype=np.float32)
        self.contact_log = np.zeros([self.num_envs, 4], dtype=np.bool)

        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

        self.multi_task = multi_task
        self.ma_env = ma_env
        self.homo_agent = homo_agent
        self.env_code = None

        self.velocity = np.zeros(self.num_envs, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action, last_action=False):
        self.wrapper.step(action, self._reward, self._done, last_action)
        return self._reward.copy(), self._done.copy()

    def change_gait(self, gait, task_idx=0):
        # print(f"change gait {gait} {task_idx}")
        self.env_code = None
        self.gait_idx = gait
        self.task_idx = task_idx
        self.wrapper.change_gait(gait)

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def reward_logging(self):
        self.wrapper.reward_logging(self.reward_log)

    def position_logging(self):
        self.wrapper.position_logging(self.position_log)
        return self.position_log

    def contact_logging(self):
        self.wrapper.contact_logging(self.contact_log)
        return self.contact_log

    def set_target_velocity(self, target_velocity):
        self.wrapper.set_target_velocity(target_velocity)

    def get_CPG_reward(self):
        self.wrapper.get_CPG_reward(self._CPG_reward)

    def get_velocity(self):
        self.wrapper.get_velocity(self.velocity)
        return self.velocity

    def ma_step(self, actions, update_statistics=True, last_action=False):
        # if last_action:
        #     print("this is last action.")
        ma_actions = actions.reshape((self.num_envs, -1))
        self.wrapper.step(ma_actions, self._reward, self._done, last_action)

        return *self.ma_observe(update_statistics), self.convert_data(self._reward.copy()), self.convert_data(self._done.copy())

    def ma_reset(self, update_statistics=False):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()
        return self.ma_observe(update_statistics)

    def ma_observe(self, update_statistics=False):
        self.wrapper.observe(self._observation, update_statistics)
        orientations, joint_angles, angular_vels, phase_ = self.split_observation_()

        if self.multi_task:
            if self.env_code is None:
                self.env_code = self.get_env_code()
                # print("env code ", self.env_code)
            env_codes = np.tile(self.env_code, self.num_envs * self.num_agents).\
                reshape((self.num_envs, *self.env_code.shape[:-1], self.num_agents, self.env_code.shape[-1]))
            actor_obs = np.concatenate([env_codes, orientations, angular_vels, phase_], axis=-1)
        else:
            actor_obs = np.concatenate([orientations, angular_vels, phase_], axis=-1)
        critic_obs = np.concatenate([orientations, angular_vels, joint_angles, phase_], axis=-1)

        # print(actor_obs[0][0])
        # input("p")
        return actor_obs, critic_obs

    def get_joint_angle(self):
        joint_angles = self._observation[:, 4:12]
        return joint_angles

    def split_observation_(self):
        orientations = np.abs(self._observation[:, 1:4]) 
        joint_angles = self._observation[:, 4:12]  
        angular_vels = np.abs(self._observation[:, 15:18]) 
        phase_ = self._observation[:, 26:34]   

        # shape (num_envs, num_agents, dim)
        orientations = np.tile(orientations, self.num_agents).reshape((*orientations.shape[:-1],
                                                                       self.num_agents, orientations.shape[-1]))
        joint_angles = joint_angles.reshape((*joint_angles.shape[:-1],
                                             self.num_agents, int(joint_angles.shape[-1] / self.num_agents)))
        angular_vels = np.tile(angular_vels, self.num_agents).reshape((*angular_vels.shape[:-1],
                                                                       self.num_agents, angular_vels.shape[-1]))
        phase_ = phase_.reshape((*phase_.shape[:-1], self.num_agents, int(phase_.shape[-1] / self.num_agents)))

        return orientations, joint_angles, angular_vels, phase_

    def get_phase(self):
        self.wrapper.get_phase(self._phase)
        return self._phase

    def get_env_code(self):
        return np.identity(self.gait_type_len)[self.task_idx]

    def convert_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data = data.reshape(*data.shape, -1)
        data = np.tile(data, self.num_agents)

        return data

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

