import numpy as np
from gym import core, spaces
import torch
import utils
from .wrapper import GroundTruthLocationEnv, ActionRepeatEnv


POINTKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', \
            'robot_global_pos', 'robot_mat']
CARKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', \
            'ballangvel_rear', 'ballquat_rear', 'robot_global_pos', 'robot_mat']
DOGGOKEYLIST = ['accelerometer', 'velocimeter', 'gyro',\
            'magnetometer', 'jointvel_hip_1_z', 'jointvel_hip_2_z', 'jointvel_hip_3_z',\
            'jointvel_hip_4_z', 'jointvel_hip_1_y', 'jointvel_hip_2_y', 'jointvel_hip_3_y',
            'jointvel_hip_4_y', 'jointvel_ankle_1', 'jointvel_ankle_2', 'jointvel_ankle_3',\
            'jointvel_ankle_4', 'jointpos_hip_1_z', 'jointpos_hip_2_z', 'jointpos_hip_3_z',\
            'jointpos_hip_4_z', 'jointpos_hip_1_y', 'jointpos_hip_2_y', 'jointpos_hip_3_y',\
            'jointpos_hip_4_y', 'jointpos_ankle_1', 'jointpos_ankle_2', 'jointpos_ankle_3', \
            'jointpos_ankle_4', 'robot_global_pos', 'robot_mat']
SAFEPOINTKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'box_lidar',\
     'hazards_lidar', 'pillars_lidar']
SAFECARKEYLIST= ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'ballangvel_rear', \
                'ballquat_rear', 'box_lidar', 'hazards_lidar', 'pillars_lidar']
class GoalFollowingAgent():
    def __init__(self, agent_name, policy=None, safe_low_level=False) -> None:
        self.policy = policy
        self.safe_low_level = safe_low_level
        if agent_name == "point" or agent_name == "mass":
            if safe_low_level:
                self.key_list = SAFEPOINTKEYLIST
            else:
                self.key_list = POINTKEYLIST
        elif agent_name == "car":
            if safe_low_level:
                self.key_list = SAFECARKEYLIST
            else:
                self.key_list = CARKEYLIST
        elif agent_name == "doggo":
            self.key_list = DOGGOKEYLIST
    def step(self, obs, ego_subgoal):
        if self.policy is None:
            action = ego_subgoal * 3
        else:
            lower_policy_obs = self.subgoal_to_input(ego_subgoal, obs)
            if self.safe_low_level:
                action = self.policy.act(lower_policy_obs)
            else:
                action = self.policy.select_action(lower_policy_obs, evaluate=True)
        return action
    def subgoal_to_input(self, subgoal, obs):
        output = []
        output.append(subgoal)
        for key in self.key_list:
            output.append(obs[key].reshape(-1))
        return np.concatenate(output)


class HierarchicalEnv(core.Env):
    " This is the environment wrapper, it wraps the safety gym environment with the MPC controller"
    def __init__(self, env, env_name, goal_following_agent, update_freq, render_flag, original_obs=False,
                force_apply=False, obs_type="MLP", node_type_flag=False, safe_low_level=False) -> None:
        super().__init__()
        self.env = env
        if not original_obs:
            self.env = GroundTruthLocationEnv(self.env, 
                                            obs_type=obs_type, 
                                            node_type_flag=node_type_flag,
                                            force_apply=force_apply)
        if obs_type == 'MLP':
            self.observation_space = self.env.observation_space
        self.original_obs = original_obs

        self.env_name = env_name
        self.goal_following_agent = goal_following_agent
        self.update_freq = update_freq
        self.render_flag = render_flag
        self.force_apply = force_apply
        # self.renderer = self.env.renderer
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_low_level = safe_low_level
    def reset(self):
        obs = self.env.reset()
        return obs
    def step(self, a):
        sum_reward = 0
        sum_cost = 0
        done_flag = False
        early_done_flag = False
        new_goal_flag = False # whether the current goal is achieved and a new goal is initialized 
        # Neither done or new goal should not be used for RL training
        for i in range(self.update_freq):
            "we need the observation and its correctponding keys"
            "because we will reorganize the obs with the subgoal"
            if self.safe_low_level:
                obs = self.env.get_lidar_obs_with_keys()
            else:
                obs = self.env.get_obj_pose()
            env_action = self.goal_following_agent.step(obs, a)
            obs, reward, done, info = self.env.step(env_action)
            sum_reward += reward
            if 'cost' in info:
                sum_cost += info['cost']
            else:
                print('cost not in the info', info)
                sum_cost += 0
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
        return obs, sum_reward, done_flag, info
