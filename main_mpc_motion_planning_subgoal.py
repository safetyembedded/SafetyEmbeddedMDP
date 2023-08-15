import os
import argparse
import time
import random
import gym
import safety_gym
import numpy as np
import itertools
import torch
import wandb
from sac_module.sac import SAC
from sac_module.replay_memory import GNNReplayMemory, ReplayMemory
from module.module import SubgoalMPCAgent, LowerLevelAgent
from module.mpc_env import SubgoalMPCEnv
import utils
from safety_gym.envs.engine import Engine
import pickle as pkl


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', type=str, default='Safexp-MassPush0-v0')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic| GNN (default: Gaussian)')
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
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--disable_cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--record_freq', type=int, default=20)
parser.add_argument('--traj_len', type=int, default=30, help='How long is the trajectory for planning')
parser.add_argument('--mpc_freq', type=int, default=10)
parser.add_argument('--update_freq', type=int, default=10,
                    help="How often do we update auxillary parameters for the motion planner")
parser.add_argument('--subgoal_thres', type=float, default=0.2)
parser.add_argument('--optimization_steps', type=int, default=10)
parser.add_argument('--predefined_layout', action="store_true", default=False)
parser.add_argument('--exp_id', type=str)
parser.add_argument('--target_weight_scale', type=float, default=100, 
                    help="the weight scale for the box and the target")
parser.add_argument('--default_init', action="store_true", default=False,
                    help="Use the default pytorch network init")
parser.add_argument('--clip_q_function', action="store_true", default=False)
parser.add_argument('--obstacle_cost_weight', type=float, default=1.0)
parser.add_argument('--disable_alias', action="store_true", default=False)
parser.add_argument('--restore_path', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument("--object_type", type=str, default='regular_box',
                    help='used for the real robot environment, choose from [regular_box, cylinder]')
parser.add_argument('--ppo_lag_low_level', action='store_true', default=False,
                    help="use the lower level agent trained by ppo lagrangian")
parser.add_argument('--gt_location', action='store_true', default=False,
                    help='whether to use ground truth location for the traj opt')
parser.add_argument('--save_replay_buffer', action='store_true', default=False)

# arguments for the pushdebug environments
parser.add_argument('--num_obstacles', type=int, default=0)
parser.add_argument('--fixed_layout', action='store_true', default=False)
parser.add_argument('--wandb_entity', type=str)

"""
TODO: for push0 environment, do we need to set the goal weight to be fixed so that the two objective won't chase each other
"""


args = parser.parse_args()

if not os.path.exists("checkpoints"):
    os.system('mkdir checkpoints')
if not os.path.exists(os.path.join("checkpoints", args.exp_id)):
    os.system(f"mkdir {os.path.join('checkpoints', args.exp_id)}")
if args.restore_path:
    restore_path = args.restore_path
    ckpt_path = args.ckpt_path
    with open(os.path.join(args.restore_path, f"args_s{args.seed}"), 'rb') as f:
        args = pkl.load(f)
    args.restore_path = restore_path
    args.ckpt_path = ckpt_path
    with open(os.path.join(args.restore_path, f"wandb_run_id_s{args.seed}"), 'rb') as f:
        WANDB_ID = pkl.load(f)
else:
    with open(os.path.join('checkpoints', args.exp_id, f"args_s{args.seed}"), 'wb') as f:
        pkl.dump(args, f)
    with open(os.path.join('checkpoints', args.exp_id, f"wandb_run_id_s{args.seed}"), 'wb') as f:
        WANDB_ID = wandb.util.generate_id()
        pkl.dump(WANDB_ID, f)
# make sure that each time the parameters update, the planner will always replan. 
assert (args.update_freq % args.mpc_freq) == 0
# Environment
# env = NormalizedActions(gym.make(args.env_name))
torch.set_default_dtype(torch.float)
if args.env == 'PushDebug':
    config = {
        'robot_base': 'xmls/mass.xml',
        'task': 'push',
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'observe_vases': False,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': args.num_obstacles,
        'vases_num': 0,
        'randomize_layout': not args.fixed_layout
    }

    env = Engine(config)
else:
    env = gym.make(args.env)
env.seed(args.seed)
env.action_space.seed(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(args.seed)
torch.random.manual_seed(args.seed)

if args.disable_alias:
    env.lidar_alias = False
# Agent
obs = env.reset()

# Set up the lower level agent
if "Mass" in args.env or args.env.lower() == 'realrobot':
    agent_name = "mass"
elif "Point" in args.env:
    agent_name = "point"
elif "Car" in args.env:
    agent_name = 'car'
elif "Doggo" in args.env:
    agent_name = 'doggo'
elif "Ant" in args.env:
    agent_name = 'ant'
if agent_name == "mass":
    lower_level_policy = None
else:
    with open(f'lower_{agent_name}.pkl', 'rb') as f:
        lower_level_policy = pkl.load(f)
lower_level_agent = LowerLevelAgent(agent_name=agent_name,
                                    subgoal_thres=args.subgoal_thres,
                                    policy=lower_level_policy)
# Set up the motion planner
total_time = 10.0
mpc_agent = SubgoalMPCAgent(lower_level_agent=lower_level_agent,
                    traj_len=args.traj_len,
                    optimization_steps=args.optimization_steps,
                    total_time=total_time,
                    mpc_freq=args.mpc_freq,
                    env_name=args.env,
                    obstacle_cost_weight=args.obstacle_cost_weight,
                    )
if args.policy == "GNN": 
    gnn_flag = True
else:
    gnn_flag = False
mpc_env = SubgoalMPCEnv(env=env,
                env_name=args.env,
                mpc_agent=mpc_agent,
                update_freq=args.update_freq,
                render_flag=False,
                gt_location=args.gt_location)
agent = SAC(num_inputs=obs.shape[0], action_space=mpc_env.action_space, args=args, default_init=args.default_init)

# wandb
wandb.init(project="safe-planning", name='{}_{}_{}_ID_{}'.format(args.policy, args.env, args.seed, args.exp_id), entity=args.wandb_entity, resume='allow', id=WANDB_ID)
if not args.restore_path:
    wandb.config.update(args)


# Memory
if args.policy == 'GNN':
    memory = GNNReplayMemory(args.replay_size, args.seed)
else:
    memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
start_time=time.time()
if args.restore_path:
    updates = agent.load_checkpoint(args.ckpt_path)
    if updates is None:
        udpates = 0

for i_episode in itertools.count(1):
    agent.train()
    episode_reward = 0
    episode_cost = 0
    episode_steps = 0
    done = False
    obs = mpc_env.reset()
    while not done:
        if args.start_steps > total_numsteps:
            action = mpc_env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(obs)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, next_q_value, q1, q2 = agent.update_parameters(memory, args.batch_size, updates)
                if updates % args.record_freq == 0:
                    current_time = time.time()
                    wandb.log({'train/critic_1_loss': critic_1_loss,
                            'train/critic_2_loss': critic_2_loss,
                            'train/policy_loss': policy_loss, 
                            'train/entropy_loss': ent_loss,
                            'train/alpha': alpha,
                            'train/next_q_vale': next_q_value,
                            'train/q1': q1,
                            'train/q2': q2,
                            'train.time_elapsed': current_time-start_time}, step=updates)
                updates += 1
        next_obs, reward, done, info = mpc_env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_cost += info['cost']

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 # there is no actual done in the safety gym. # NOTE: if there is an actual done, please modify this. 
        # mask = 1 if episode_steps == max_episode_steps else float(not done)
        if (not info['early_done']) and (not info['new_goal']): 
            memory.push(obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        mask=mask) # Append transition to memory
        obs = next_obs
    if (i_episode % args.save_freq)==0:
        if args.save_replay_buffer:
            save_item = memory
        else:
            save_item = None
        agent.save_checkpoint(args.env, suffix='{}_s{}'.format(i_episode, args.seed), folder_name=args.exp_id, updates=updates, replay_buffer=save_item)

    if total_numsteps > args.num_steps:
        break
    wandb.log({'train/reward': episode_reward, }, step=updates)
    wandb.log({'train/cost': episode_cost, }, step=updates)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}, cost:{round(episode_cost, 2)}")
    agent.eval()
    if i_episode % 20 == 0 and args.eval is True:
        avg_reward = 0.
        avg_cost = 0.
        episodes = 10
        for _ in range(episodes):
            obs = mpc_env.reset()
            episode_reward = 0
            episode_cost = 0
            done = False
            while not done:
                action = agent.select_action(obs, evaluate=True)  # Sample action from policy

                next_obs, reward, done, info = mpc_env.step(action)
                episode_reward += reward
                episode_cost += info['cost']
                obs = next_obs
            avg_reward += episode_reward
            avg_cost += episode_cost
        avg_reward /= episodes
        avg_cost /= episodes

        wandb.log({'test/reward': avg_reward}, step=updates)
        wandb.log({'test/cost': avg_cost}, step=updates)

        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}, Avg. Cost: {round(avg_cost, 2)}")
        print("----------------------------------------")

# env.close()

