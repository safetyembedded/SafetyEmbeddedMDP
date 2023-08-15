import argparse
import itertools
import gym
import safety_gym
import numpy as np
import torch
import wandb
from sac_module.sac import SAC
import time
from sac_module.replay_memory import ReplayMemory
from module.wrapper import ActionRepeatEnv, GroundTruthLocationEnv
from safety_gym.envs.engine import Engine

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
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
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
parser.add_argument('--exp_id', type=str)
parser.add_argument('--near_goal', action="store_true", default=False)
parser.add_argument('--clip_q_function', action="store_true", default=False)
parser.add_argument('--wandb_entity', type=str)

args = parser.parse_args()

# Environment
if args.agent.lower() == 'doggo':
    goal_range = 0.6
    robot_keepout = 0.4
else:
    goal_range = 0.4
    robot_keepout = 0.2
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
        'goal_size': 0.05,
    }
if args.near_goal:
    config.update({"near_goal": True,
        "goal_range": goal_range,
        "goal_keepout": 0.0,
        'robot_keepout': robot_keepout,
        "_seed": args.seed,})
        
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
agent = SAC(num_inputs=state.shape[0], action_space=env.action_space, args=args, default_init=False)
#Tesnorboard
wandb.init(project="safe-planning", name=f'lower_{args.agent}_{args.seed}_ID_{args.exp_id}', entity=args.wandb_entity)
wandb.config.update(args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
start_time = time.time()
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_cost = 0
    episode_steps = 0
    done = False
    obs = env.reset()
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
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
        next_obs, reward, done, info = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_cost += info['cost']

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.num_steps else float(not done)
        # mask = 1
        # NOTE: the environment is never done in safety gym, if we want to modify to other environment. Pay attention to this
        memory.push(obs, action, reward, next_obs, mask) # Append transition to memory
        obs = next_obs
    if (i_episode % args.save_freq)==0:
        agent.save_checkpoint(args.agent, suffix='{}_s{}'.format(i_episode, args.seed), folder_name=args.exp_id)

    if total_numsteps > args.num_steps:
        break

    wandb.log({'train/reward': episode_reward, }, step=updates)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}, cost: {round(episode_cost, 2)}")

    if i_episode % 20 == 0 and args.eval is True:
        episodes = 10
        reward_list = []
        cost_list = []
        for _  in range(episodes):
            obs = env.reset()

            episode_reward = 0
            episode_cost = 0
            done = False
            while not done:
                action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_cost += info['cost']

                obs = next_obs

            reward_list.append(episode_reward)
            cost_list.append(episode_cost)


        avg_reward = np.array(reward_list).mean()
        std_reward = np.array(reward_list).std()
        avg_cost = np.array(cost_list).mean()
        std_cost = np.array(cost_list).std()

        wandb.log({'test/reward': avg_reward}, step=updates)

        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Avg. Reward: {avg_reward} Std. reward: {std_reward}, Avg. Cost: {avg_cost} Std. Cost:{std_cost}")
        print("----------------------------------------")

env.close()

