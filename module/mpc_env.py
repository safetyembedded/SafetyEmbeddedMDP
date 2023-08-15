import numpy as np
from gym import core, spaces
import torch
import utils
import copy


class SubgoalMPCEnv(core.Env):
    " This is the environment wrapper, it wraps the safety gym environment with the MPC controller"
    def __init__(self, env, env_name, mpc_agent, update_freq, render_flag,
                plot_freq=0, video_writer=None,
                subgoal_scale=2.0, record_traj=False, gt_location=False) -> None:
        super().__init__()
        self.env = env
        self.env_name = env_name
        self.mpc_agent = mpc_agent
        self.update_freq = update_freq
        self.step_ctr = 0
        self.render_flag = render_flag
        self.plot_freq = plot_freq
        self.video_writer = video_writer
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.subgoal_scale = subgoal_scale
        self.record_traj = record_traj # whether to record the data and use for plotting later
        self.gt_location = gt_location
        if record_traj:
            self.data_buf = []
    def reset(self):
        self.step_ctr = 0
        obs = self.env.reset()
        self.mpc_agent.step_counter = 0
        self.data_buf = []
        return obs
    def step(self, a):
        self.mpc_agent.subgoal = torch.tensor(a * self.subgoal_scale).view(2).float()
        sum_reward = 0
        sum_cost = 0
        done_flag = False
        early_done_flag = False
        new_goal_flag = False # whether the current goal is achieved and a new goal is initialized 
        # Neither done or new goal should not be used for RL training
        for i in range(self.update_freq):
            obj_pose = self.env.get_obj_pose()
            lidar_obs = self.env.get_lidar_obs_with_keys()
            # obs.update(obj_pose)
            lidar_obs.update({'robot_global_pos': obj_pose['robot_global_pos'], 'robot_mat': obj_pose['robot_mat']})
            self.get_pose_from_lidar(lidar_obs)
            if (self.mpc_agent.step_counter % self.mpc_agent.mpc_freq) == 0:
                lidar_obs_for_plot = lidar_obs
            if self.gt_location:
                obs = obj_pose
            else:
                obs = lidar_obs
            env_action = self.mpc_agent.step(obs)
            # if (self.mpc_agent.step_counter % self.mpc_agent.mpc_freq) == 0:
            lidar_obs_for_plot.update({'goal_pos': a * self.subgoal_scale})
            rl_obs, reward, done, info = self.env.step(env_action)
            sum_reward += reward
            if 'cost' in info:
                sum_cost += info['cost']
            else:
                print('cost not in the info', info)
                sum_cost += 0
            self.step_ctr += 1
            traj = self.mpc_agent.lower_level_agent.traj
            if self.record_traj:
                self.data_buf.append({'obj_pose': obj_pose,
                'traj': traj, 
                'passed_subgoal_index': self.mpc_agent.lower_level_agent.passed_subgoal_index,
                'reward': reward,
                'cost': info['cost'],
                'sum_cost': sum_cost,
                'sum_reward': sum_reward,
                'lidar_obs': copy.deepcopy(lidar_obs_for_plot)})
            if self.plot_freq > 0 and self.video_writer is not None:
                if (self.step_ctr % self.plot_freq) == 0:
                    # NOTE: afeter plotting, we should never use the obs or lidar obs again, since it has been updated
                    if 'Push' in self.env_name:
                        utils.plot_traj(obj_pose, traj,
                        self.mpc_agent.lower_level_agent.passed_subgoal_index,
                        reward=reward,
                        cost=info['cost'],
                        video_writer=self.video_writer,
                        lidar_obs=copy.deepcopy(lidar_obs_for_plot))
                    else:
                        utils.plot_traj(obj_pose, traj, self.mpc_agent.lower_level_agent.passed_subgoal_index,
                        reward=reward,
                        cost=info['cost'],
                        video_writer=self.video_writer,
                        lidar_obs=copy.deepcopy(lidar_obs_for_plot))

            
            if self.render_flag:
                self.env.render()
            if 'goal_met' in info.keys() and (i != self.update_freq-1):
                new_goal_flag = True
                # break
            if done:
                done_flag = True
                if i != self.update_freq-1:
                    early_done_flag = True
                break # if it is done, we terminate the simulation
            if early_done_flag:
                assert done_flag
        info['early_done'] = early_done_flag
        info['new_goal'] = new_goal_flag
        info['cost'] = sum_cost
        return rl_obs, sum_reward, done_flag, info
    def get_pose_from_lidar(self, obs):
        # have double checked this aligns with the original safety gym gt location
        lidar_max_dist = self.env.lidar_max_dist
        bin_size = (np.pi * 2) / self.env.lidar_num_bins
        pos_est = {}
        # sensor reading = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
        for key in obs.keys():
            if 'lidar' in key:
                obj_name = key.split('_lidar')[0]
                result_list = []
                for i, reading in enumerate(obs[key]):
                    if reading > 0:
                        angle = bin_size * (i + 0.5)
                        distance = (1-reading) * lidar_max_dist
                        x = distance * np.cos(angle)
                        y = distance * np.sin(angle)
                        result_list.append([x, y])
                pos_est.update({f'{obj_name}_pos': np.array(result_list)})
        obs.update(pos_est)
        return obs