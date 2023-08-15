import random
import os
import argparse
import gym
import safety_gym
import numpy as np
import torch
from sac_module.sac import SAC
from module.module import SubgoalMPCAgent, LowerLevelAgent
from module.mpc_env import SubgoalMPCEnv
import utils
import imageio
from safety_gym.envs.engine import Engine
import tqdm
import pickle as pkl



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', type=str, default='Safexp-MassPush1-v0')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
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
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--record_freq', type=int, default=10)
parser.add_argument('--traj_len', type=int, default=30, help='How long is the trajectory for planning')
parser.add_argument('--num_episodes', type=int, default=10, help='Num of episodes')
parser.add_argument('--mpc_freq', type=int, default=10)
parser.add_argument('--update_freq', type=int, default=10,
                    help="How often do we update auxillary parameters for the motion planner")
parser.add_argument('--subgoal_thres', type=float, default=0.2)
parser.add_argument('--optimization_steps', type=int, default=5)
parser.add_argument('--plot_freq', type=int, default=0, help='plotting freq')
parser.add_argument('--predefined_layout', action="store_true", default=False)
parser.add_argument('--fixed_goal_weight', action='store_true', default=False)
parser.add_argument('--method', type=str, default='mpc',
                    help='choose from [mpc, sac]')
parser.add_argument('--force_apply', action='store_true', 
                    help='apply push0 policy to push1 environment')
parser.add_argument('--straight_init', action='store_true', default=False)
parser.add_argument("--disable_viz", action="store_true", default=False)

# arguments for the pushdebug environments
parser.add_argument('--num_obstacles', type=int, default=0)
parser.add_argument('--fixed_layout', action='store_true', default=False)

# arguments for sac baseline
parser.add_argument('--num_action_repeat', type=int, default=0)
parser.add_argument('--target_weight_scale', type=float, default=100, 
                    help="the weight scale for the box and the target")

parser.add_argument('--obstacle_cost_weight', type=float, default=1.0)
parser.add_argument('--default_init', action='store_true', default=False)
parser.add_argument('--clip_q_function', action="store_true", default=False)
parser.add_argument('--disable_lagrangian', action="store_true", default=False)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--max_distance', type=float, default=0.5, 
                    help='the threshold used for obstacle costs')
parser.add_argument('--single_goal', action="store_true", default=False)

parser.add_argument('--record_traj', action='store_true', default=False)
parser.add_argument('--add_lidar_error', action='store_true', default=False)


"""
TODO: for push0 environment, do we need to set the goal weight to be fixed so that the two objective won't chase each other
"""


args = parser.parse_args()
if not os.path.exists('temp'):
    os.system('mkdir temp')
if not os.path.exists('temp/pkl'):
    os.system('mkdir temp/pkl')

# make sure that each time the parameters update, the planner will always replan. 
assert (args.update_freq % args.mpc_freq) == 0
# Environment
# env = NormalizedActions(gym.make(args.env_name))
torch.set_default_dtype(torch.float)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(args.seed)
torch.random.manual_seed(args.seed)
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
env.observation_space.seed(args.seed)

if args.single_goal:
    env.continue_goal = False
obs = env.reset()

# Set up the lower level agent
if "Mass" in args.env:
    agent_name = "mass"
elif "Point" in args.env:
    agent_name = "point"
elif "Car" in args.env:
    agent_name = 'car'
elif "Doggo" in args.env:
    agent_name = 'doggo'
elif 'Ant' in args.env:
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
if not args.disable_lagrangian:
    test_flag = True
else:
    test_flag = False
mpc_agent = SubgoalMPCAgent(lower_level_agent=lower_level_agent,
                    traj_len=args.traj_len,
                    optimization_steps=args.optimization_steps,
                    total_time=total_time,
                    mpc_freq=args.mpc_freq,
                    env_name=args.env,
                    obstacle_cost_weight=args.obstacle_cost_weight,
                    lagrangian_test=test_flag,
                    max_distance=args.max_distance,
                    add_lidar_error=args.add_lidar_error
                    )
if args.policy == "GNN": 
    gnn_flag = True
else:
    gnn_flag = False
if args.plot_freq > 0:
    render_flag = False
else:
    render_flag = True
    # render_flag = False
if args.disable_viz:
    render_flag = False
env = SubgoalMPCEnv(env=env,
                env_name=args.env,
                mpc_agent=mpc_agent,
                update_freq=args.update_freq,
                render_flag=render_flag,
                plot_freq=args.plot_freq,
                record_traj=args.record_traj)
agent = SAC(num_inputs=obs.shape[0], action_space=env.action_space, args=args, default_init=args.default_init)
if args.load_dir is None:
    print("error, not loading any checkpoint")
else:
    agent.load_checkpoint(args.load_dir)
total_numsteps = 0
with torch.no_grad():
    # env.mpc_agent.set_lambda(32)
    reward_list = []
    cost_list = []
    num_success = 0
    for i_episode in tqdm.tqdm(range(args.num_episodes)):
        if args.plot_freq > 0 and args.method == 'mpc':
            video_writer = imageio.get_writer(f"temp/video_mpc{i_episode}_s{args.seed}.mp4", fps=10)
            env.video_writer = video_writer
        obs = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_step = 0
        done = False
        step = 0
        while not done:
            action = agent.select_action(obs, evaluate=True)  # Sample action from policy
            next_obs, reward, done, info = env.step(action)
            # print(reward)
            # print(episode_step, reward, action[0])
            if 'goal_met' in info and info['goal_met'] == True:
                num_success += 1
            if args.method == 'sac':
                env.render()
            
            episode_step += 1
            episode_reward += reward
            episode_cost += info['cost']
            obs = next_obs
        print("Reward: {}, Cost: {}".format(episode_reward, episode_cost))
        reward_list.append(episode_reward)
        cost_list.append(episode_cost)

        if args.record_traj:
            with open (f'temp/pkl/traj_{i_episode}_s{args.seed}', 'wb') as f:
                pkl.dump(env.data_buf, f)
                env.data_buf = []

            
        if args.method == 'mpc':
            if env.video_writer is not None:
                env.video_writer.close()
    avg_reward = np.array(reward_list).mean()
    std_reward = np.array(reward_list).std()
    avg_cost = np.array(cost_list).mean()
    std_cost = np.array(cost_list).std()


    print("----------------------------------------")
    print(f"Test Episodes: {args.num_episodes}, Avg. Reward: {avg_reward} Std. reward: {std_reward}, Avg. Cost: {avg_cost} Std. Cost:{std_cost}")
    print("----------------------------------------")
    if args.single_goal:
        print(f'success rate: {num_success / args.num_episodes}')

    # env.close()

