import math
import torch
import numpy as np
from torch_geometric.utils import remove_self_loops

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_edge_index_complete_graph(num_nodes):
    temp = torch.arange(num_nodes).long()
    node_j = temp.repeat(num_nodes)
    node_i = temp.repeat_interleave(num_nodes)
    edge_index = torch.stack([node_i, node_j], dim=0)
    edge_index = remove_self_loops(edge_index)
    edge_index = edge_index[0]
    return edge_index
def get_edge_index_agent_centric(num_nodes, node_type):
    "assume that the agent node type is labeled as 0"
    "only have edges from other nodes to the agent node"
    "this function never works for batch data, which means num_nodes should be a scalar"
    agent_index, = np.where(node_type == 0)
    node_i = torch.arange(num_nodes)
    node_j = torch.tensor(agent_index).repeat(num_nodes)
    edge_index = torch.stack([node_i, node_j])
    edge_index = remove_self_loops(edge_index)
    edge_index = edge_index[0]
    return edge_index
