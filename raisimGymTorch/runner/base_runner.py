import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.task_idx = 0

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.all_args.env_name = self.envs.gait_names

        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.group_num = 4
        self.policy_idx = [i for i in range(4)]
        if self.all_args.homo_agent:
            self.group_num = 2
            self.policy_idx = [0, 0, 1, 1]

        # parameters
        self.env_name = [self.all_args.env_name] if isinstance(self.all_args.env_name, str) else self.all_args.env_name

        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.update_statistics = self.all_args.update_statistics

        self.n_rollout_threads = self.envs.n_rollout_threads

        # self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.thresh = self.all_args.thresh

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.eval_episodes = self.all_args.eval_episodes

        self.max_eval_reward = -np.inf

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', os.path.split(config["run_dir"])[-1])

        print("==="*10)
        print("env_name:", self.env_name)
        print("load model dir:", self.model_dir)
        print("save model dir:", config["run_dir"])
        print("run dir:", self.run_dir)
        print("env config file:", f"{self.all_args.env_config}.yaml")
        if self.all_args.homo_agent:
            print("homogenous agent group num:", self.group_num)
        print("==="*10)

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.writter = SummaryWriter(logdir=self.run_dir, comment="_{}".format(self.env_name[0]) if len(self.env_name)==1 else "_multi_task")
            self.save_dir = config["run_dir"]
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                os.makedirs(os.path.join(self.save_dir, "latest"))

        # TODO  define trainer
        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        self.policy = []

        for agent_id in range(self.group_num):
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        self.envs.share_observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args,
                           self.policy[self.policy_idx[agent_id]],
                           device=self.device,
                           use_adaptive_learning_rate=self.all_args.use_adaptive_learning_rate)
            # buffer
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       self.envs.share_observation_space[agent_id],
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1],
                                                                  task_idx=self.task_idx)
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], task_idx=self.task_idx)
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, save_idx=0, latest=False):
        for agent_id in range(self.group_num):  # self.num_agents
            policy_actor = self.trainer[agent_id].policy.actor
            if latest:
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "latest/actor_agent" + str(agent_id) + ".pt")
            else:
                torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/model_{save_idx}" + "/actor_agent" + str(agent_id) + ".pt")

            for task_idx in range(len(self.env_name)):
                policy_critic = self.trainer[agent_id].policy.critics[task_idx]
                torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/model_{save_idx}" + "/critic_agent" + str(agent_id) + "task_" + str(task_idx) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            for task_idx in range(len(self.env_name)):
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + "task_" + str(task_idx) + '.pt')
                self.policy[agent_id].critics[task_idx].load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
