import gym
from gym.spaces import Box
import numpy as np
import utils
class GroundTruthLocationEnv(gym.ObservationWrapper):
    def __init__(self, env, obs_type, node_type_flag, force_apply):
        "node type flag: whether input the node type as an one-hot encoding"
        "force_apply, for MLP agent, if we want to apply Push0 agent to Push1"
        "we omit some of the observations"
        super().__init__(env)
        self.obs_type = obs_type.upper()
        self.node_type_flag = node_type_flag
        self.force_apply = force_apply
        env.reset()
        if self.obs_type == 'GNN':
            output = env.get_gnn_agent_obs(node_type_flag=self.node_type_flag)
        elif self.obs_type == 'MLP':
            output = env.get_mlp_agent_obs(force_apply=self.force_apply)
            self.observation_space = Box(shape=output.shape, low=-np.inf, high=np.inf)
        elif self.obs_type == "raw":
            output = env.obs()
            self.observation_space = Box(shape=output.shape, low=-np.inf, high=np.inf)
        
    def observation(self, obs):
        if self.obs_type == 'GNN':
            output = self.env.get_gnn_agent_obs(node_type_flag=self.node_type_flag)
        elif self.obs_type == 'MLP':
            output = self.env.get_mlp_agent_obs(force_apply=self.force_apply)
        elif self.obs_type == "raw":
            output = self.env.observation(obs)
        else:
            raise ValueError("obs type should choose from [GNN, MLP, raw]")
        return output

class ActionRepeatEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, num_action_repeat) -> None:
        super().__init__(env)
        self.num_action_repeat = num_action_repeat
    def step(self, act):
        sum_reward = 0
        sum_cost = 0
        for i in range(self.num_action_repeat):
            next_obs, reward, done, info = super().step(act)
            sum_reward += reward
            sum_cost += info['cost']
        info = {'cost': sum_cost}
        return next_obs, sum_reward, done, info


class ObstacleInRewardEnv(gym.Wrapper):
    def __init__(self, env, max_distance, obstacle_weight):
        super().__init__(env)
        self.max_distance = max_distance
        self.obstacle_weight = obstacle_weight
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['original_reward'] = reward
        obj_pose = self.env.get_obj_pose() # in ego centric coordinate
        obstacle_pos = utils.get_obstacle_array(obj_pose)
        obstacle_cost = self.obstacle_cost(np.array([[0,0]]), obstacle_pos)
        assert self.env.ego_centric
        reward -= self.obstacle_weight * obstacle_cost.sum()
        return obs, reward, done, info
    # def reward(self, reward):
    #     obj_pose = self.env.get_obj_pose() # in ego centric coordinate
    #     obstacle_pos = utils.get_obstacle_array(obj_pose)
    #     assert self.env.ego_centric
    #     obstacle_cost = self.obstacle_cost(np.array([[0,0]]), obstacle_pos)
    #     return reward - self.obstacle_weight * obstacle_cost.sum()
    def obstacle_cost(self, agent_pos, obstacle_pos):
        dist2obstacle = self.max_distance - np.linalg.norm(agent_pos - obstacle_pos, axis=-1)
        dist2obstacle = np.clip(dist2obstacle, 0, np.inf)
        obstacle_cost = np.power(dist2obstacle, 2)
        return obstacle_cost


class GoalGroundTruthLocationEnv(gym.ObservationWrapper):
    "only goal was used as a groundtruth environment, the other observation was kept as is"
    "this is used to train the low level agent"
    def __init__(self, env):
        "node type flag: whether input the node type as an one-hot encoding"
        "force_apply, for MLP agent, if we want to apply Push0 agent to Push1"
        "we omit some of the observations"
        super().__init__(env)
        env.reset()
        self.obs_name_dict = env.obs_space_dict
        output = self.observation(None)
        self.observation_space = Box(shape=output.shape, low=-np.inf, high=np.inf)

    def observation(self, obs):
        obs_dict = self.env.get_lidar_obs_with_keys()
        output = []
        for key in self.obs_name_dict:
            if 'goal' in key.lower():
                continue
            output.append(obs_dict[key].reshape(-1))
        output.append(self.env.ego_xy(self.env.goal_pos[:2]))
        output = np.concatenate(output)
        return output
