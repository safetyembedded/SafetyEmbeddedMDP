from email.policy import default
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .sac_utils import soft_update, hard_update, get_edge_index_agent_centric
from .model import GNNQNetwork, GNNGaussianPolicy, GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, action_space, args, default_init, **kwargs):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = not args.disable_automatic_entropy_tuning
        try:
            self.clip_q_function = args.clip_q_function
        except AttributeError:
            self.clip_q_function = False

        self.device = torch.device("cuda" if not args.disable_cuda else "cpu")


        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(kwargs['num_inputs'], action_space.shape[0], args.hidden_size, default_init, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.critic = QNetwork(kwargs['num_inputs'], action_space.shape[0], args.hidden_size, default_init=default_init, clip_q_function=False).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

            self.critic_target = QNetwork(kwargs['num_inputs'], action_space.shape[0], args.hidden_size, default_init=default_init, clip_q_function=self.clip_q_function).to(self.device)
            hard_update(self.critic_target, self.critic)

        elif self.policy_type == "GNN":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = GNNGaussianPolicy(in_channels=kwargs['in_channels'],
                                            state_dim=kwargs['state_dim'],
                                            num_actions=action_space.shape[0],
                                            hidden_dim=args.hidden_size,
                                            num_types=kwargs['num_types'],
                                            default_init=default_init,
                                            action_space=action_space).to(device=self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.critic = GNNQNetwork(in_channels=kwargs['in_channels'],
                                      state_dim=kwargs['state_dim'],
                                      num_actions=action_space.shape[0],
                                      hidden_dim=args.hidden_size,
                                      default_init=default_init,
                                      num_types=kwargs['num_types'],
                                      clip_q_function=False).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
            self.critic_target = GNNQNetwork(in_channels=kwargs['in_channels'],
                                      state_dim=kwargs['state_dim'],
                                      num_actions=action_space.shape[0],
                                      hidden_dim=args.hidden_size,
                                      default_init=default_init,
                                      num_types=kwargs['num_types'],
                                      clip_q_function=self.clip_q_function).to(device=self.device)
            hard_update(self.critic_target, self.critic)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs=kwargs['num_inputs'],
                                              num_actions=action_space.shape[0],
                                              hidden_dim=args.hidden_size,
                                              default_init=default_init,
                                              action_space=action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.critic = QNetwork(kwargs['num_inputs'], action_space.shape[0], args.hidden_size, default_init=default_init, clip_q_function=False).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

            self.critic_target = QNetwork(kwargs['num_inputs'], action_space.shape[0], args.hidden_size, default_init=default_init, clip_q_function=self.clip_q_function).to(self.device)
            hard_update(self.critic_target, self.critic)


    def select_action(self, obs, evaluate=False):
        "only use for evaluation, which means doesn't support batch operation"
        if self.policy_type == 'GNN':
            node, node_type, state = obs
            edge_index = get_edge_index_agent_centric(node.shape[0], node_type)
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            node = torch.FloatTensor(node).to(self.device)
            edge_index = torch.LongTensor(edge_index).to(self.device)
            node_type = torch.LongTensor(node_type).to(self.device)
            node_num = torch.LongTensor([node.shape[0]]).to(self.device)
            obs = node, state, edge_index, node_type, node_num
        else:
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(obs)
        else:
            _, _, action = self.policy.sample(obs)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        if self.policy_type == "GNN":
            node_batch, state_batch, action_batch, reward_batch, next_node_batch, next_state_batch, mask_batch, node_type_batch, edge_index_batch, node_num = memory.sample(batch_size=batch_size)
            node_batch = torch.FloatTensor(node_batch).to(self.device)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)            
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            next_node_batch = torch.FloatTensor(next_node_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            node_type_batch = torch.LongTensor(node_type_batch).to(self.device)
            edge_index_batch = torch.LongTensor(edge_index_batch).to(self.device)
            node_num = torch.LongTensor(node_num).to(self.device)
            with torch.no_grad():
                next_obs = (next_node_batch, next_state_batch, edge_index_batch, node_type_batch, node_num)
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_obs)
                qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            obs = (node_batch, state_batch, edge_index_batch, node_type_batch, node_num)
            qf1, qf2 = self.critic(obs, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.policy_type == "GNN":
            obs = (node_batch, state_batch, edge_index_batch, node_type_batch, node_num)
            pi, log_pi, _ = self.policy.sample(obs)
            qf1_pi, qf2_pi = self.critic(obs, pi)
        else:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), next_q_value.mean().item(), qf1.mean().item(), qf2.mean().item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None, folder_name=None, replay_buffer=None, updates=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if folder_name is not None:
            if not os.path.exists(os.path.join('checkpoints', folder_name)):
                os,os.makedirs(os.path.join('checkpoints', folder_name))
        if ckpt_path is None:
            if folder_name is not None:
                ckpt_path = os.path.join("checkpoints",folder_name, "sac_checkpoint_{}_{}".format(env_name, suffix))
            else:
                ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'updates': updates}, ckpt_path)
        if replay_buffer is not None:
            seed = int(suffix.split('s')[-1])
            torch.save(replay_buffer, f'replay_buffer_s{seed}')

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            try:
                updates = checkpoint['updates']
            except KeyError:
                updates = None
            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
        return updates
    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()
    def train(self):
        self.policy.train()
        self.critic.train()
        self.critic_target.train()

