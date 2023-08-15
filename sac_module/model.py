import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
MAX_Q = 50

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.hidden_channels = hidden_channels
        
        self.encoding_mlp = Seq(Linear(in_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels))
        self.mlp1 = Seq(Linear(2 * hidden_channels, hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, hidden_channels))
                       
        self.mlp2 = Seq(Linear(hidden_channels, hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index, node_type):
        "input info mation into a MessagePassing layer"
        "x: [N, in_channels]"
        "edge_index [2, E]"
        "node_type denotes the type of each node"
        "agent denotes which one is the agent itself, which will be used as the output node"
        x = self.encoding_mlp(x)
        x = self.propagate(edge_index, x=x)
        agent_index, = torch.where(node_type == 0)
        x = torch.index_select(x, dim=0, index=agent_index)
        x = self.mlp2(x)
        return x

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        
        return self.mlp1(tmp)




class GNNQNetwork(nn.Module):
    def __init__(self, in_channels, state_dim, num_actions, hidden_dim, num_types, default_init=False, clip_q_function=False):
        super(GNNQNetwork, self).__init__()

        # Q1 architecture
        # self.gnn_1 = GNN(in_channels, hidden_dim, hidden_dim)
        # self.mlp_1 = Seq(Linear(hidden_dim + state_dim + num_actions, hidden_dim),
        #                ReLU(),
        #                Linear(hidden_dim, hidden_dim),
        #                ReLU(),
        #                Linear(hidden_dim, 1))

        # # Q2 architecture
        # self.gnn_2 = GNN(in_channels, hidden_dim, hidden_dim)
        # self.mlp_2 = Seq(Linear(hidden_dim + state_dim + num_actions, hidden_dim),
        #                ReLU(),
        #                Linear(hidden_dim, hidden_dim),
        #                ReLU(),
        #                Linear(hidden_dim, 1))

        # Q1 architecture
        self.gnn_1 = GNN(in_channels + state_dim + num_actions, hidden_dim, hidden_dim)
        self.mlp_1 = Seq(Linear(hidden_dim, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, 1))

        # Q2 architecture
        self.gnn_2 = GNN(in_channels + state_dim + num_actions, hidden_dim, hidden_dim)
        self.mlp_2 = Seq(Linear(hidden_dim, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, 1))
        if not default_init:
            self.apply(weights_init_)
        self.clip_q_function = clip_q_function

    def forward(self, obs, action):
        node, robot_state, edge_index, node_type, node_num = obs
        xu = torch.cat([robot_state, action], 1)

        # gnn_hidden_1 = self.gnn_1.forward(node, edge_index, node_type)
        # x1 = torch.cat((gnn_hidden_1, xu), dim=1)
        # x1 = self.mlp_1(x1)

        # gnn_hidden_2 = self.gnn_2.forward(node, edge_index, node_type)
        # x2 = torch.cat((gnn_hidden_2, xu), dim=1)
        # x2 = self.mlp_2(x2)

        # the following code concatenate the state with the node
        xu = xu.repeat_interleave(node_num, dim=0)
        node = torch.cat((node, xu), dim=1)

        gnn_hidden_1 = self.gnn_1.forward(node, edge_index, node_type)
        x1 = self.mlp_1(gnn_hidden_1)

        gnn_hidden_2 = self.gnn_2.forward(node, edge_index, node_type)
        x2 = self.mlp_2(gnn_hidden_2)        
        
        if self.clip_q_function:
            x1 = torch.min(x1, torch.tensor(MAX_Q))
            x2 = torch.min(x2, torch.tensor(MAX_Q))
        return x1, x2


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, default_init, clip_q_function=False):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        if not default_init:
            self.apply(weights_init_)
        self.clip_q_function = clip_q_function

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        if self.clip_q_function:
            x1 = torch.min(x1, torch.tensor(MAX_Q))
            x2 = torch.min(x2, torch.tensor(MAX_Q))

        return x1, x2
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, default_init, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        if not default_init:
            self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class GNNGaussianPolicy(nn.Module):
    def __init__(self, in_channels, state_dim, num_actions, hidden_dim, num_types, action_space=None, default_init=False):
        super(GNNGaussianPolicy, self).__init__()
        
        # self.gnn = GNN(in_channels, hidden_dim, hidden_dim)
        # self.mlp = Seq(Linear(hidden_dim + state_dim, hidden_dim),
        #                ReLU(),
        #                Linear(hidden_dim, hidden_dim),
        #                ReLU())
        self.gnn = GNN(in_channels + state_dim, hidden_dim, hidden_dim)
        self.mlp = Seq(Linear(hidden_dim, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, hidden_dim),
                       ReLU())

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        if not default_init:
            self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def _forward(self, obs):
        # node num is not the scalar telling how many nodes we have in this graph,
        # but an vector indicating how many nodes are there in a single data frame
        node, robot_state, edge_index, node_type, node_num = obs
        # gnn_hidden = self.gnn.forward(node, edge_index, node_type)
        # x = torch.cat((gnn_hidden, robot_state), dim=1)
        # x = self.mlp(x)
        robot_state = robot_state.repeat_interleave(node_num, dim=0)
        node = torch.cat((node, robot_state), dim=1)
        gnn_hidden = self.gnn.forward(node, edge_index, node_type)
        x = self.mlp(gnn_hidden)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self._forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GNNGaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, default_init, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        if not default_init:
            self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
