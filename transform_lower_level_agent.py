import argparse
import gym
import safety_gym
import numpy as np
import torch
from sac_module.sac import SAC
from sac_module.replay_memory import ReplayMemory
from module.wrapper import ActionRepeatEnv, GroundTruthLocationEnv
from safety_gym.envs.engine import Engine
import pickle as pkl
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--agent', default="point",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--disable_automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--disable_cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--record_freq', type=int, default=20)
parser.add_argument('--load_path', type=str)
args = parser.parse_args()

# Environment
config = {
        'robot_base': f'xmls/{args.agent.lower()}.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        # 'observe_hazards': True,
        # 'observe_vases': False,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 0,
        'vases_num': 0,
        'randomize_layout': True,
        "near_goal": True,
        "goal_range": 0.4,
        "goal_keepout": 0.0,
        'goal_size': 0.05,
        'robot_keepout': 0.2,
        "_seed": args.seed,
    }

env = Engine(config)
env.seed(args.seed)
env.action_space.seed(args.seed)

obs_type = "MLP"
env = GroundTruthLocationEnv(env=env,
                             obs_type=obs_type,
                             node_type_flag=True,
                             force_apply=False)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
state = env.reset()
agent = SAC(num_inputs=state.shape[0], action_space=env.action_space, args=args, default_init=True)
agent.load_checkpoint(args.load_path)
with open(f'lower_{args.agent.lower()}.pkl','wb') as f:
    pkl.dump(agent, f)
#Tesnorboard

