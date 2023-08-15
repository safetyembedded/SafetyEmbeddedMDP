import random
import numpy as np
import os 
import pickle
from .sac_utils import get_edge_index_agent_centric
import copy

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, mask = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, mask

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = f"checkpoints/sac_buffer_{env_name}_{suffix}"
        print(f'Saving buffer to {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print(f'Loading buffer from {save_path}')

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class GNNReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs, mask):
        node, node_type, state = obs
        next_node, next_node_type, next_state = next_obs
        assert (node_type == next_node_type).all()
        edge_index = get_edge_index_agent_centric(node_type.shape[0], node_type)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (node, state, action, reward, next_node, next_state, mask, node_type, edge_index)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        node, state, action, reward, next_node, next_state, mask, node_type, edge_index = zip(*batch)
        edge_index = copy.deepcopy(list(edge_index))
        displacement = 0
        node_num_list = []
        for i,_ in enumerate(edge_index):
            num_node = edge_index[i].max().item() + 1
            edge_index[i] += displacement
            displacement += num_node
            # the maximum index + 1 should be the num of nodes, we assume there is no isolated node
            node_num_list.append(num_node)
        node = np.concatenate(node)
        state = np.stack(state)
        action = np.stack(action)
        reward = np.array(reward)
        next_node = np.concatenate(next_node)
        next_state = np.stack(next_state)
        mask = np.array(mask)
        node_type = np.concatenate(node_type)
        edge_index = np.concatenate(edge_index, axis=1)
        node_num = np.array(node_num_list)
        return node, state, action, reward, next_node, next_state, mask, node_type, edge_index, node_num

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = f"checkpoints/sac_buffer_{env_name}_{suffix}"
        print(f'Saving buffer to {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print(f'Loading buffer from {save_path}')

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

