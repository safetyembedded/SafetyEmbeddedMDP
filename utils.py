import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.spatial.transform import Rotation

def init_theseus():
    temp = torch.eye(3).cuda()
    try:
        torch.linalg.cholesky(temp)
    except:
        pass
def get_trajectory(values_dict, traj_len):
    "transform the trajectory from theseus output to an array"
    trajectory = torch.empty(values_dict[f"pose_0"].shape[0], 4, traj_len)
    for i in range(traj_len):
        trajectory[:, :2, i] = values_dict[f"pose_{i}"]
        trajectory[:, 2:, i] = values_dict[f"vel_{i}"]
    return trajectory

def ego_xy(robot_global_pos, robot_mat, pos):
    ''' Return the egocentric XY vector to a position from the robot '''
    assert pos.shape == (2,), f'Bad pos {pos}'

    orientation = Rotation.from_matrix(robot_mat)
    fixed_frame_euler =  orientation.as_euler('xyz')
    fixed_frame_euler[0] = 0
    fixed_frame_euler[1] = 0
    robot_mat = Rotation.from_euler('xyz', fixed_frame_euler).as_matrix()

    pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
    world_3vec = pos_3vec - robot_global_pos
    return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates
def global_xy(robot_global_pos, robot_mat, pos):
    assert pos.shape == (2,), f'Bad pos {pos}'

    orientation = Rotation.from_matrix(robot_mat)
    fixed_frame_euler =  orientation.as_euler('xyz')
    fixed_frame_euler[0] = 0
    fixed_frame_euler[1] = 0
    robot_mat = Rotation.from_euler('xyz', fixed_frame_euler).as_matrix()

    pos_3vec = np.concatenate([pos, [0]])
    world_3vec = np.matmul(pos_3vec, robot_mat.transpose())
    world_3vec += robot_global_pos
    return world_3vec[:2]
def ego_v_xy(robot_global_pos, robot_mat, v):
    ''' Return the egocentric XY velocity to a position from the robot '''
    assert v.shape == (2,), f'Bad velocity {v}'

    orientation = Rotation.from_matrix(robot_mat)
    fixed_frame_euler =  orientation.as_euler('xyz')
    fixed_frame_euler[0] = 0
    fixed_frame_euler[1] = 0
    robot_mat = Rotation.from_euler('xyz', fixed_frame_euler).as_matrix()
    v_3vec = np.concatenate([v, [0]])  # Add a zero z-coordinate
    return np.matmul(v_3vec, robot_mat)[:2]  # only take XY coordinates
def global_v_xy(robot_global_pos, robot_mat, v):
    assert v.shape == (2,), f'Bad v {v}'

    orientation = Rotation.from_matrix(robot_mat)
    fixed_frame_euler =  orientation.as_euler('xyz')
    fixed_frame_euler[0] = 0
    fixed_frame_euler[1] = 0
    robot_mat = Rotation.from_euler('xyz', fixed_frame_euler).as_matrix()

    v_3vec = np.concatenate([v, [0]])
    world_3vec = np.matmul(v_3vec, robot_mat.transpose())
    return world_3vec[:2]

def get_obstacle_array(obs):
    "extract obstacle array from the observation"
    obstacle_list = []
    if 'hazards_pos' in obs.keys():
        if obs['hazards_pos'].size != 0:
            obstacle_list.append(obs['hazards_pos'])
    if 'vases_pos' in obs.keys():
        if obs['vases_pos'].size != 0:
            obstacle_list.append(obs['vases_pos'])
    if 'pillars_pos' in obs.keys():
        if obs['pillars_pos'].size != 0:        
            obstacle_list.append(obs['pillars_pos'])
    if 'gremlins_pos' in obs.keys():
        if obs['gremlins_pos'].size != 0:
            obstacle_list.append(obs['gremlins_pos'])
    if len(obstacle_list) > 0:
        obstacle_list = np.concatenate(obstacle_list)
    return obstacle_list

def plot_traj(obs, traj, passed_subgoal_index, reward=None, cost=None, video_writer=None, box_bias=None, box_weight=None, goal_weight=None, lidar_obs=None, save_folder='temp'): 
    obstacle_radius = 0.3
    fig, axes = plt.subplots()     
      
    robot_global_pos = obs['robot_global_pos']
    robot_mat = obs['robot_mat']
    goal_pos = obs['goal_pos']
    goal_pos = global_xy(robot_global_pos, robot_mat, goal_pos)
    subgoal_pos = traj[0,:2, passed_subgoal_index + 1]
    
    if 'hazards_pos' in obs:
        obstacle_pos = obs['hazards_pos']
        for i,_ in enumerate(obstacle_pos):
            obstacle_pos[i] = global_xy(robot_global_pos, robot_mat, obstacle_pos[i])
            circle = plt.Circle(obstacle_pos[i], obstacle_radius, fill=False, color='r')
            axes.add_artist(circle)
            if lidar_obs is None:
                circle = plt.Circle(obstacle_pos[i], 0.5, fill=False, color='k')
                axes.add_artist(circle)
        axes.scatter(obstacle_pos[:,0], obstacle_pos[:,1], c='coral', label='hazard')
    
    if 'vases_pos' in obs:
        vases_pos = obs['vases_pos']
        for i,_ in enumerate(vases_pos):
            vases_pos[i] = global_xy(robot_global_pos, robot_mat, vases_pos[i])
            circle = plt.Circle(vases_pos[i], obstacle_radius, fill=False, color='r')
            axes.add_artist(circle)
        axes.scatter(vases_pos[:,0], vases_pos[:,1], c='tomato', label='vase')
    
    if 'pillars_pos' in obs:
        pillars_pos = obs['pillars_pos']
        for i,_ in enumerate(pillars_pos):
            pillars_pos[i] = global_xy(robot_global_pos, robot_mat, pillars_pos[i])
            circle = plt.Circle(pillars_pos[i], obstacle_radius, fill=False, color='r')
            axes.add_artist(circle)
            if lidar_obs is None:
                circle = plt.Circle(pillars_pos[i], 0.5, fill=False, color='k')
                axes.add_artist(circle)

        axes.scatter(pillars_pos[:,0], pillars_pos[:,1], c='gray', label='pillar')

    if 'box_pos' in obs:
        box_pos = global_xy(robot_global_pos, robot_mat, obs['box_pos'])
        if box_bias is not None:
            box_pos_plus_bias = global_xy(robot_global_pos, robot_mat, obs['box_pos'] + box_bias)
        else:
            box_pos_plus_bias = global_xy(robot_global_pos, robot_mat, obs['box_pos'])
        if box_weight is not None:
            circle_box = plt.Circle(box_pos, pow(box_weight,2)/5000, fill=False)
            axes.add_artist(circle_box)
        if goal_weight is not None:
            circle_goal = plt.Circle(goal_pos, pow(goal_weight,2)/5000, fill=False)
            axes.add_artist(circle_goal)
        axes.scatter(box_pos[0], box_pos[1], c='purple', label='box')
        if box_bias is not None:
            axes.scatter(box_pos_plus_bias[0], box_pos_plus_bias[1], c='violet', label='box+bias')
        axes.plot([box_pos[0], box_pos_plus_bias[0]], [box_pos[1], box_pos_plus_bias[1]], linestyle='--')

    traj = traj[0,:2,:]
    axes.scatter(traj[0,:], traj[1,:], label='waypont')
    axes.scatter(robot_global_pos[0], robot_global_pos[1], c='lime', label='agent')
    axes.scatter(goal_pos[0], goal_pos[1], color='r', label='goal')
    axes.scatter(subgoal_pos[0], subgoal_pos[1], c='orange', label='subgoal', marker='+')

    if lidar_obs is not None:
        robot_global_pos = lidar_obs['robot_global_pos']
        robot_mat = lidar_obs['robot_mat']
        if 'hazards_pos' in lidar_obs:
            obstacle_pos = lidar_obs['hazards_pos']
            for i,_ in enumerate(obstacle_pos):
                obstacle_pos[i] = global_xy(robot_global_pos, robot_mat, obstacle_pos[i])
                circle = plt.Circle(obstacle_pos[i], 0.05, fill=False, color='coral')
                axes.add_artist(circle)
                circle = plt.Circle(obstacle_pos[i], 0.5, fill=False, color='k')
                axes.add_artist(circle)
        if 'vases_pos' in lidar_obs:
            vases_pos = lidar_obs['vases_pos']
            for i,_ in enumerate(vases_pos):
                vases_pos[i] = global_xy(robot_global_pos, robot_mat, vases_pos[i])
                circle = plt.Circle(vases_pos[i], 0.05, fill=False, color='tomato')
                axes.add_artist(circle)
                circle = plt.Circle(vases_pos[i], 0.5, fill=False, color='k')
                axes.add_artist(circle)
        if 'pillars_pos' in lidar_obs:
            pillars_pos = lidar_obs['pillars_pos']
            for i,_ in enumerate(pillars_pos):
                pillars_pos[i] = global_xy(robot_global_pos, robot_mat, pillars_pos[i])
                circle = plt.Circle(pillars_pos[i], 0.05, fill=False, color='gray')
                axes.add_artist(circle)
                circle = plt.Circle(pillars_pos[i], 0.5, fill=False, color='k')
                axes.add_artist(circle)

        rl_goal = lidar_obs['goal_pos']
        rl_goal = global_xy(robot_global_pos, robot_mat, rl_goal)
        axes.scatter(rl_goal[0], rl_goal[1], c='red', label='rl-subgoal', marker='*')
        # circle = plt.Circle(rl_goal, 0.08, fill=False, color='red', )
        # axes.add_artist(circle)


        
    plt.legend()
    # plt.xlim(-2.2,2.2)
    # plt.ylim(-2.2,2.2)
    plt.xlim(-2.2,6.2)
    plt.ylim(-4.2,4.2)
    # if not os.path.exists('temp'):
    #     os.mkdir('temp')
    if reward is not None:
        plt.text(-1.9, 1.8, "Reward: {}".format(reward), fontsize='large')
    if cost is not None:
        plt.text(-1.9, 1.6, "Cost: {}".format(cost), fontsize='large')
    # plt.show()
    plt.savefig(os.path.join(save_folder, 'temp.png'))
    plt.clf()
    plt.cla()
    if video_writer is not None:
        video_writer.append_data(imageio.imread(os.path.join(save_folder, 'temp.png')))


def min_dist2hazards(robot_pos, hazards_pos):
    dist_list = []
    for hazard_pos in hazards_pos:
        dist = np.linalg.norm(robot_pos - hazard_pos)
        dist_list.append(dist)
    return min(dist_list)
