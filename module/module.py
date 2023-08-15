import theseus as th
import torch
import numpy as np
from .cost_function import ObstacleCost, ObstacleCostWithVelocity
import utils

POINTKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', \
            'robot_global_pos', 'robot_mat']
CARKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', \
            'ballangvel_rear', 'ballquat_rear', 'robot_global_pos', 'robot_mat']
DOGGOKEYLIST = ['accelerometer', 'velocimeter', 'gyro',\
            'magnetometer', 'jointvel_hip_1_z', 'jointvel_hip_2_z', 'jointvel_hip_3_z',\
            'jointvel_hip_4_z', 'jointvel_hip_1_y', 'jointvel_hip_2_y', 'jointvel_hip_3_y',
            'jointvel_hip_4_y', 'jointvel_ankle_1', 'jointvel_ankle_2', 'jointvel_ankle_3',\
            'jointvel_ankle_4', 'jointpos_hip_1_z', 'jointpos_hip_2_z', 'jointpos_hip_3_z',\
            'jointpos_hip_4_z', 'jointpos_hip_1_y', 'jointpos_hip_2_y', 'jointpos_hip_3_y',\
            'jointpos_hip_4_y', 'jointpos_ankle_1', 'jointpos_ankle_2', 'jointpos_ankle_3', \
            'jointpos_ankle_4', 'robot_global_pos', 'robot_mat']
ANTKEYLIST = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'hip_1_vel', \
                'ankle_1_vel', 'hip_2_vel', 'ankle_2_vel', 'hip_3_vel', 'ankle_3_vel', 'hip_4_vel',\
                'ankle_4_vel', 'hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3',\
                'hip_4', 'ankle_4', 'robot_global_pos', 'robot_mat']


# LIDAR_NUM = 16
LIDAR_NUM = 30

MAX_LAMBDA=1024

def point_diff_func(optim_vars, aux_vars):
    current_pos = optim_vars[0]
    target_pos, weight, bias = aux_vars
    error = current_pos.tensor - (target_pos.tensor + bias.tensor)
    error = error * weight.tensor
    return error
class TheseusMotionPlanner:
    def __init__(self,
                traj_len,
                num_obstacle,
                optimization_steps,
                total_time=10.0,
                env_name=None,
                straight_init=False,
                max_distance=0.5,
                obstacle_cost_weight=1.0) -> None:
        utils.init_theseus()
        self.prev_traj = None # Note that it is in the global frame
        self.prev_goal = None
        # self.prev_agent_pose = None
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        "Set up planning parameters"
        self.traj_len = traj_len
        num_time_steps = traj_len - 1
        self.total_time = total_time
        self.env_name = env_name
        self.straight_init = straight_init
        dt_val = total_time / num_time_steps
        Qc_inv = [[1.0, 0.0], [0.0, 1.0]]
        # Qc_inv = [[1.0, 0.0], [0.0, 100.0]]
        boundary_w = 100.0

        poses = []
        velocities = []
        for i in range(traj_len):
            poses.append(th.Point2(name=f"pose_{i}", dtype=torch.float))
            velocities.append(th.Point2(name=f"vel_{i}", dtype=torch.float))
        start_point = th.Point2(name="start", dtype=torch.float)
        goal_point = th.Point2(name="goal", dtype=torch.float) 
        if 'Push' in env_name:
            box_point = th.Point2(name='box')

        dt = th.Variable(torch.tensor(dt_val).float().view(1, 1), name="dt")
        # Cost weight to use for all GP-dynamics cost functions
        gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv).float(), dt)
        # boundary_w = th.Vector(1, name="boundary_w")
        boundary_cost_weight = th.ScaleCostWeight(boundary_w)

        objective = th.Objective(dtype=torch.float)

        # Fixed starting position
        objective.add(
            th.Difference(poses[0], start_point, boundary_cost_weight, name="pose_0")
        )
        # Fixed initial velocity
        # objective.add(
        #     th.Difference(
        #         velocities[0],
        #         th.Point2(tensor=torch.zeros(1, 2)),
        #         boundary_cost_weight,
        #         name="vel_0",
        #     )
        # )
        if 'Push' in env_name:
            goal_weight = th.Vector(1, name="goal_weight")
            optim_vars = [poses[-1]]
            # TODO: check if the zeros tensor will be updated if we use a differentiable trajectory optimizer
            aux_vars = goal_point, goal_weight, th.Variable(tensor=torch.zeros((1,2)).float())
            goal_cost_function = th.AutoDiffCostFunction(optim_vars,
                                                    point_diff_func,
                                                    dim=2,
                                                    cost_weight=th.ScaleCostWeight(1.),
                                                    aux_vars=aux_vars,
                                                    name="pose_N")
            objective.add(goal_cost_function)

            box_weight = th.Vector(1, name="box_weight")
            box_bias = th.Vector(2, name="box_bias")
            optim_vars = [poses[-1]]
            aux_vars = box_point, box_weight, box_bias
            box_cost_function = th.AutoDiffCostFunction(optim_vars,
                                                    point_diff_func,
                                                    dim=2,
                                                    cost_weight=th.ScaleCostWeight(1.),
                                                    aux_vars=aux_vars,
                                                    name="pose_N_box")
            objective.add(box_cost_function)

            # box_repulsion_weight = th.Vector(1, name='box_repulsion_weight')
            # optim_vars = [poses[-1]]
            # aux_vars = box_point, box_repulsion_weight, th.Variable(tensor=torch.zeros((1,2)).float())
            # box_repulsion_cost_function = th.AutoDiffCostFunction(optim_vars,
            #                                         point_diff_func,
            #                                         dim=2,
            #                                         cost_weight=th.ScaleCostWeight(1.),
            #                                         aux_vars=aux_vars,
            #                                         name="pose_N_box_repulsion")
            # objective.add(box_repulsion_cost_function)
        else:
            objective.add(
                th.Difference(
                    poses[-1], goal_point, boundary_cost_weight, name="pose_N"
                )
            )


        obstacle_pos_list = []
        for i in range(num_obstacle):
            # obstacle_pos_list.append(th.Point2(torch.tensor(pos).float().to(device), name=f'obstacle_{i}'))
            obstacle_pos_list.append(th.Point2(name=f'obstacle_{i}'))

        obstacle_cost_weight = th.ScaleCostWeight(obstacle_cost_weight)
        for i, obstacle_pos in enumerate(obstacle_pos_list):
            for j, pose in enumerate(poses):
                objective.add(ObstacleCost(cost_weight=obstacle_cost_weight, 
                                           agent_pos=pose, 
                                           obstacle_pos=obstacle_pos, 
                                           max_distance=max_distance, 
                                           name=f'agent_{j}_obstacle_{i}'))
                # objective.add(ObstacleCostWithVelocity(cost_weight=obstacle_cost_weight, 
                #                             agent_pos=pose, 
                #                             agent_v=velocities[j],
                #                             obstacle_pos=obstacle_pos, 
                #                             max_distance=max_distance, 
                #                             name=f'agent_{j}_obstacle_{i}'))
        for i in range(1, traj_len):
            objective.add(
                (
                    th.eb.GPMotionModel(
                        poses[i - 1],
                        velocities[i - 1],
                        poses[i],
                        velocities[i],
                        dt,
                        gp_cost_weight,
                        name=f"gp_{i}",
                    )
                )
            )


        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=optimization_steps,
            step_size=1.0,
            abs_err_tolerance=1e-3,
            rel_err_tolerance=5e-2,
        )
        self.motion_planner = th.TheseusLayer(optimizer)
    def plan(self, start, goal, obstacles, robot_global_pos, robot_mat, **kwargs):
        "Only the trajectory stored is in a global coordinate, everything else is in the ego frame"
        "Including the input"
        "Note that only works for single batch right now"
        start = torch.tensor(start).unsqueeze(0).float()
        goal = torch.tensor(goal).unsqueeze(0).float()
        input_dict = {"start":start.to(self.device),
                    "goal": goal.to(self.device)}
        if self.prev_traj is None or self.straight_init:
            if 'Push' in self.env_name:
                # box = torch.tensor(box).unsqueeze(0).float().to(self.device)
                goal_weight = kwargs['goal_weight'].item()
                box_weight = kwargs['box_weight'].item()
                if goal_weight > box_weight:
                    init_target = goal
                else:
                    init_target = kwargs['box'] + kwargs['box_bias']
                input_dict.update(self.get_straight_line_inputs(start, init_target))
            else:
                input_dict.update(self.get_straight_line_inputs(start, goal))
                # input_dict.update(self.get_starter_point_init(start))
        else:
            ego_traj = self.global2ego(self.prev_traj, robot_global_pos, robot_mat)
            # update the prev_traj, set the closest point and its previous points to the current pos
            distance_list = []
            for i in range(self.traj_len):
                distance_list.append((start.cpu() - ego_traj[f'pose_{i}']).norm()) 
            distance_list = torch.tensor(distance_list)
            min_index, = torch.where(distance_list == distance_list.min())
            if len(min_index) > 0:
                min_index = min_index[0].item()
            else:
                min_index = min_index.item()
            for i in range(min_index + 1):
                ego_traj[f'pose_{i}'] = start.cpu()
            for key in ego_traj.keys():
                ego_traj[key] = ego_traj[key].to(self.device)
            input_dict.update(ego_traj)
        for i, pos in enumerate(obstacles):
            input_dict.update({f'obstacle_{i}': torch.tensor(pos).unsqueeze(0).float().to(self.device)})  
        # input specific to each environment
        if "Push" in self.env_name:
            for key in kwargs.keys():
                kwargs[key] = kwargs[key].to(self.device)
            input_dict.update(kwargs)
        with torch.no_grad():
            final_values, info = self.motion_planner.forward(
                input_dict,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": False,
                    "damping": 0.1,

                }
            )
        global_traj = self.ego2global(info.best_solution, robot_global_pos, robot_mat)
        if kwargs['lagrangian_test']:
            ego_traj = utils.get_trajectory(info.best_solution, self.traj_len)
            dist = self.traj_dist_to_obstacle(traj=ego_traj[0,:2,:].numpy().T, obstacles=obstacles)
            info.min_dist = dist.min()
        # all the trajectory output outside of the class are within global coordinate
        self.prev_traj = global_traj
        return global_traj

    def traj_dist_to_obstacle(self, traj, obstacles):
        # all are in the ego frame
        num_obstacles = obstacles.shape[0]
        obstacles = np.tile(obstacles, (self.traj_len, 1))
        traj = np.repeat(traj, repeats=num_obstacles, axis=0)
        dist = np.linalg.norm(traj-obstacles, axis=1)
        
        return dist

    def get_straight_line_inputs(self, start, goal):
        # Returns a dictionary with pose and velocity variable names associated to a 
        # straight line trajectory between start and goal
        start_goal_dist = goal - start
        avg_vel = start_goal_dist / self.total_time
        unit_trajectory_len = start_goal_dist / (self.traj_len - 1)
        input_dict = {}
        for i in range(self.traj_len):
            input_dict[f"pose_{i}"] = (start + unit_trajectory_len * i).to(self.device)
            if i == 0 or i == self.traj_len - 1:
                input_dict[f"vel_{i}"] = torch.zeros_like(avg_vel).to(self.device)
            else:
                input_dict[f"vel_{i}"] = avg_vel.to(self.device)
        return input_dict
    def get_starter_point_init(self, start):
        input_dict = {}
        for i in range(self.traj_len):
            input_dict[f"pose_{i}"] = start.to(self.device)
            input_dict[f"vel_{i}"] = torch.zeros_like(start).to(self.device)
        return input_dict

    def ego2global(self, traj, robot_global_pos, robot_mat):
        "transform the trajectory from ego coordinate to global coordinate"
        for key in traj.keys():
            if 'pos' in key:
                global_pos = utils.global_xy(robot_global_pos,
                                             robot_mat,
                                             traj[key].squeeze(0).detach().cpu().numpy())
                traj[key] = torch.FloatTensor(global_pos).unsqueeze(0)
            elif 'vel' in key:
                global_vel = utils.global_v_xy(robot_global_pos,
                                               robot_mat,
                                               traj[key].squeeze(0).detach().cpu().numpy())
                traj[key] = torch.FloatTensor(global_vel).unsqueeze(0)
        return traj
    def global2ego(self, traj, robot_global_pos, robot_mat):
        "transform the trajectory from global coordinate to ego coordinate "
        for key in traj.keys():
            if 'pos' in key:
                global_pos = utils.ego_xy(robot_global_pos,
                                          robot_mat,
                                          traj[key].squeeze(0).detach().cpu().numpy())
                traj[key] = torch.FloatTensor(global_pos).unsqueeze(0)
            elif 'vel' in key:
                global_vel = utils.ego_v_xy(robot_global_pos,
                                            robot_mat,
                                            traj[key].squeeze(0).detach().cpu().numpy())
                traj[key] = torch.FloatTensor(global_vel).unsqueeze(0)
        return traj


class MPCAgent(TheseusMotionPlanner):
    "the agent replans every k timesteps, if it is not time for replanning, it just follows the subgoal"
    def __init__(self,
                lower_level_agent,
                traj_len,
                num_obstacle,
                optimization_steps,
                total_time,
                mpc_freq,
                env_name,
                straight_init,
                lagrangian_test=False,
                max_distance=0.5,
                obstacle_cost_weight=1.0) -> None:
        super().__init__(traj_len,
            num_obstacle,
            optimization_steps,
            total_time=total_time,
            env_name=env_name,
            straight_init=straight_init,
            max_distance=max_distance,
            obstacle_cost_weight=obstacle_cost_weight)
        self.motion_planner.to(device=self.device, dtype=torch.float)
        self.lower_level_agent = lower_level_agent
        self.mpc_freq = mpc_freq
        self.step_counter = 0
        self.lagragian_test = lagrangian_test
        self.max_obstacle_distance = max_distance
        self.init_obstacle_lambda = obstacle_cost_weight
        if "Push" in env_name:
            self.goal_weight = torch.zeros((1,1)).float()
            self.box_weight = torch.zeros((1,1)).float()
            self.box_bias = torch.zeros((1,2)).float()
    def step(self, obs):
        "only works for 1 batch size right now"
        robot_global_pos = obs['robot_global_pos']
        robot_mat = obs['robot_mat']

        goal_changed = False

        goal_world_frame = utils.global_xy(robot_global_pos, robot_mat, obs['goal_pos'])
        if self.prev_goal is not None:
            if np.linalg.norm(goal_world_frame - self.prev_goal) > 0.0001:
                self.lower_level_agent.passed_subgoal_index = 1
                self.lower_level_agent.traj = None
                self.prev_traj = None
                goal_changed = True
        self.prev_goal = goal_world_frame
        obstacle_pos = utils.get_obstacle_array(obs)
        agent_pos = np.array([0,0])
        goal_pos = obs['goal_pos']
        try:
            dist2obstacles = np.linalg.norm((obstacle_pos - agent_pos), axis=1).min()
        except ValueError:
            dist2obstacles = 999
        if ((self.step_counter % self.mpc_freq) == 0) or goal_changed:
            "the trajectory here is in a global coordinate, but the input are in the ego coordinate"
            
            if 'Push' in self.env_name:
                box = torch.FloatTensor(obs['box_pos']).unsqueeze(0)
                # global_raw_traj = self.plan(start=agent_pos,
                #                      goal=goal_pos,
                #                      obstacles=obstacle_pos,
                #                      robot_global_pos=robot_global_pos,
                #                      robot_mat=robot_mat,
                #                      goal_weight=self.goal_weight,
                #                      box_weight=self.box_weight,
                #                      box_bias=self.box_bias,
                #                      box=box,
                #                      )
                if self.lagrangian_test:
                    if dist2obstacles < self.max_obstacle_distance:
                        self.set_lambda(MAX_LAMBDA)
                    else:
                        self.set_lambda(self.init_obstacle_lambda)
                    unsafe_plan = True
                    while unsafe_plan:
                        global_raw_traj, info = self.plan(start=agent_pos,
                                                    goal=goal_pos,
                                                    obstacles=obstacle_pos,
                                                    robot_global_pos=robot_global_pos,
                                                    robot_mat=robot_mat,
                                                    goal_weight=self.goal_weight,
                                                    box_weight=self.box_weight,
                                                    box_bias=self.box_bias,
                                                    box=box,
                                                    lagrangian_test=self.lagragian_test
                                                    )
                        min_dist = info.min_dist
                        if min_dist < self.max_obstacle_distance: # given that the distance for actual cost is 0.2
                            unsafe_plan = True
                            lagrangian_value = 2 * self.obstacle_lambda
                            # print(lagrangian_value)
                            self.set_lambda(lagrangian_value)
                            if lagrangian_value > MAX_LAMBDA: # to avoid being stuck at this point, which never gets a safe trajectory
                                unsafe_plan = False
                            # print(f"unsafe_plan, set lambda to be {self.obstacle_lambda}")
                        else:
                            unsafe_plan = False
                else:
                    global_raw_traj, info = self.plan(start=agent_pos,
                                                    goal=goal_pos,
                                                    obstacles=obstacle_pos,
                                                    robot_global_pos=robot_global_pos,
                                                    robot_mat=robot_mat,
                                                    goal_weight=self.goal_weight,
                                                    box_weight=self.box_weight,
                                                    box_bias=self.box_bias,
                                                    box=box,
                                                    lagrangian_test=self.lagragian_test
                                                    )
            else: 
                global_raw_traj = self.plan(start=agent_pos,
                                     goal=goal_pos,
                                     obstacles=obstacle_pos,
                                     robot_global_pos=robot_global_pos,
                                     robot_mat=robot_mat)
            global_traj = utils.get_trajectory(global_raw_traj, self.traj_len)
            self.lower_level_agent.update_traj(global_traj)
        action = self.lower_level_agent.step(obs)
        # print(time.time() - start_time)

        self.step_counter += 1
        return action
    def update_learnable_params(self, **kwargs):
        "Update the learnable parameters to the motion planner"
        if "Push" in self.env_name:
            self.goal_weight = kwargs['goal_weight']
            self.box_weight = kwargs['box_weight']
            self.box_bias = kwargs['box_bias']
            
    def set_lambda(self, obstacle_lambda):
        obstacle_lambda = min(obstacle_lambda, MAX_LAMBDA)
        super().__init__(traj_len=self.traj_len,
                      optimization_steps=self.optimization_steps,
                      total_time=self.total_time,
                      env_name=self.env_name,
                      max_distance=self.max_distance,
                      obstacle_cost_weight=self.obstacle_cost_weight,
                      obstacle_lambda=obstacle_lambda,
                    #   lidar_error=self.lidar_error
                      )
        self.motion_planner.to(device=self.device, dtype=torch.float)


class SubgoalTheseusMotionPlanner(TheseusMotionPlanner):
    def __init__(self, traj_len, optimization_steps, total_time=10, env_name=None, max_distance=0.5, obstacle_cost_weight=1, obstacle_lambda=1) -> None:
        utils.init_theseus()
        self.obstacle_lambda = obstacle_lambda
        self.prev_traj = None # Note that it is in the global frame
        self.prev_goal = None
        # self.prev_agent_pose = None
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        "Set up planning parameters"
        self.traj_len = traj_len
        self.optimization_steps = optimization_steps
        num_time_steps = traj_len - 1
        self.total_time = total_time
        self.env_name = env_name
        self.max_distance = max_distance
        self.obstacle_cost_weight = obstacle_cost_weight
        # self.lidar_error = lidar_error
        dt_val = total_time / num_time_steps
        Qc_inv = [[1.0, 0.0], [0.0, 1.0]]
        # Qc_inv = [[1.0, 0.0], [0.0, 100.0]]

        poses = []
        velocities = []
        for i in range(traj_len):
            poses.append(th.Point2(name=f"pose_{i}", dtype=torch.float))
            velocities.append(th.Point2(name=f"vel_{i}", dtype=torch.float))
        start_point = th.Point2(name="start", dtype=torch.float)
        goal_point = th.Vector(2, name="goal")
        if 'Push' in env_name:
            box_point = th.Point2(name='box')

        dt = th.Variable(torch.tensor(dt_val).float().view(1, 1), name="dt")
        # Cost weight to use for all GP-dynamics cost functions
        gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv).float() * obstacle_lambda * 5, dt)
        start_boundary_cost_weight = th.ScaleCostWeight(100.0 * obstacle_lambda)
        end_boundary_cost_weight = th.ScaleCostWeight(1.0)

        objective = th.Objective(dtype=torch.float)

        # Fixed starting position
        objective.add(
            th.Difference(poses[0], start_point, start_boundary_cost_weight, name="pose_0")
        )
        objective.add(
            th.Difference(
                velocities[0],
                th.Point2(tensor=torch.zeros(1, 2)),
                start_boundary_cost_weight,
                name="vel_0",
            )
        )
        # objective.add(
        #     th.Difference(
        #         velocities[-1],
        #         th.Point2(tensor=torch.zeros(1, 2)),
        #         start_boundary_cost_weight,
        #         name="vel_N",
        #     )
        # )
        # goal_weight = th.Vector(1, name="goal_weight")
        # optim_vars = [poses[-1]]

        # if this one doesn't work, choose the previous function
        objective.add(
            th.Difference(
                poses[-1], goal_point, end_boundary_cost_weight, name="pose_N"
            )
        )
        # objective.cost_functions['pose_N'].weight = th.ScaleCostWeight(0)
        # breakpoint()
        obstacle_pos_list = []
        for i in range(LIDAR_NUM):
            # obstacle_pos_list.append(th.Point2(torch.tensor(pos).float().to(device), name=f'obstacle_{i}'))
            obstacle_pos_list.append(th.Point2(name=f'obstacle_{i}'))

        obstacle_cost_weight = th.ScaleCostWeight(obstacle_cost_weight * obstacle_lambda)
        for i, obstacle_pos in enumerate(obstacle_pos_list):
            for j, pose in enumerate(poses):
                # if i < len(lidar_error):
                #     objective.add(ObstacleCost(cost_weight=obstacle_cost_weight, 
                #                             agent_pos=pose, 
                #                             obstacle_pos=obstacle_pos, 
                #                             max_distance=max_distance+lidar_error[i], 
                #                             name=f'agent_{j}_obstacle_{i}'))
                # else:
                objective.add(ObstacleCost(cost_weight=obstacle_cost_weight, 
                                        agent_pos=pose, 
                                        obstacle_pos=obstacle_pos, 
                                        max_distance=max_distance, 
                                        name=f'agent_{j}_obstacle_{i}'))
                # objective.add(ObstacleCostWithVelocity(cost_weight=obstacle_cost_weight, 
                #                             agent_pos=pose, 
                #                             agent_v=velocities[j],
                #                             obstacle_pos=obstacle_pos, 
                #                             max_distance=max_distance, 
                #                             name=f'agent_{j}_obstacle_{i}'))
        for i in range(1, traj_len):
            objective.add(
                (
                    th.eb.GPMotionModel(
                        poses[i - 1],
                        velocities[i - 1],
                        poses[i],
                        velocities[i],
                        dt,
                        gp_cost_weight,
                        name=f"gp_{i}",
                    )
                )
            )

        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=optimization_steps,
            step_size=1.0,
            abs_err_tolerance=1e-3,
            rel_err_tolerance=5e-2,
        )
        self.motion_planner = th.TheseusLayer(optimizer)
    def plan(self, start, subgoal, obstacles, robot_global_pos, robot_mat, **kwargs):
        "everything else is in the ego frame"
        "Including the input"
        "Note that only works for single batch right now"
        start = torch.tensor(start).unsqueeze(0).float()
        subgoal = torch.tensor(subgoal).unsqueeze(0).float()
        input_dict = {"start":start.to(self.device),
                    "goal": subgoal.to(self.device)}
        # input_dict.update(self.get_straight_line_inputs(start, subgoal))
        input_dict.update(self.get_starter_point_init(start))

        if len(obstacles) > 0:
            obstacles = np.concatenate((obstacles, np.ones((LIDAR_NUM - len(obstacles), 2)) * 999))
        else:
            obstacles = np.ones((LIDAR_NUM, 2)) * 999
        assert len(obstacles) == LIDAR_NUM
        for i, pos in enumerate(obstacles):
            input_dict.update({f'obstacle_{i}': torch.tensor(pos).unsqueeze(0).float().to(self.device)})  
        # input specific to each environment
        with torch.no_grad():
            final_values, info = self.motion_planner.forward(
                input_dict,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": False,
                    "damping": 0.1,

                }
            )
        if kwargs['lagrangian_test']:
            ego_traj = utils.get_trajectory(info.best_solution, self.traj_len)
            dist = self.traj_dist_to_obstacle(traj=ego_traj[0,:2,:].numpy().T, obstacles=obstacles)
            info.min_dist = dist.min()
        global_traj = self.ego2global(info.best_solution, robot_global_pos, robot_mat)
        # all the trajectory output outside of the class are within global coordinate
        return global_traj, info
    def traj_dist_to_obstacle(self, traj, obstacles):
        # all are in the ego frame
        num_obstacles = obstacles.shape[0]
        obstacles = np.tile(obstacles, (self.traj_len, 1))
        traj = np.repeat(traj, repeats=num_obstacles, axis=0)
        dist = np.linalg.norm(traj-obstacles, axis=1)
        
        return dist
    

class SubgoalMPCAgent(SubgoalTheseusMotionPlanner):
    "the agent replans every k timesteps, if it is not time for replanning, it just follows the subgoal"
    def __init__(self,
                lower_level_agent,
                traj_len,
                optimization_steps,
                total_time,
                mpc_freq,
                env_name,
                max_distance=0.5,
                obstacle_cost_weight=1.0,
                lagrangian_test=False,
                init_obstacle_lambda=1.0,
                add_lidar_error=False) -> None:   # if test = True, we will compute the distance to obstacles
        super().__init__(traj_len,
            optimization_steps,
            total_time=total_time,
            env_name=env_name,
            max_distance=max_distance + 0.01,
            obstacle_cost_weight=obstacle_cost_weight,
            obstacle_lambda=init_obstacle_lambda)
        self.motion_planner.to(device=self.device, dtype=torch.float)
        self.lower_level_agent = lower_level_agent
        self.mpc_freq = mpc_freq
        self.lagrangian_test = lagrangian_test
        self.max_obstacle_distance = max_distance
        self.step_counter = 0
        self.subgoal = torch.zeros((2)).float()
        self.plan_flag = False # indicate whether the MPC agent plans in this time step
        self.init_obstacle_lambda = init_obstacle_lambda
        self.add_lidar_error = add_lidar_error
        # self.unsafe_plan = False
    def step(self, obs):
        "everything is in the robot frame right now"
        self.plan_flag = False
        # self.unsafe_plan = False
        robot_global_pos = obs['robot_global_pos']
        robot_mat = obs['robot_mat']

        goal_changed = False

        obstacle_pos = utils.get_obstacle_array(obs)
        agent_pos = np.array([0,0])
        try:
            dist2obstacles = np.linalg.norm((obstacle_pos - agent_pos), axis=1).min()
        except ValueError:
            dist2obstacles = 999
            
        if ((self.step_counter % self.mpc_freq) == 0) or goal_changed:
            "the trajectory here is in a global coordinate, but the input are in the ego coordinate"
            self.plan_flag = True
            if self.lagrangian_test:
                if dist2obstacles < self.max_obstacle_distance:
                    self.set_lambda(MAX_LAMBDA)
                else:
                    self.set_lambda(self.init_obstacle_lambda)
                unsafe_plan = True
                while unsafe_plan:
                    # if self.add_lidar_error:
                    #     lidar_error = self.get_lidar_error(obstacle_pos)
                    #     self.set_lidar_error(lidar_error)
                    global_raw_traj, info = self.plan(start=agent_pos,
                                            subgoal=self.subgoal,
                                            obstacles=obstacle_pos,
                                            robot_global_pos=robot_global_pos,
                                            robot_mat=robot_mat,
                                            lagrangian_test=self.lagrangian_test)
                    min_dist = info.min_dist
                    if min_dist < self.max_obstacle_distance: # given that the distance for actual cost is 0.2
                        unsafe_plan = True
                        lagrangian_value = 2 * self.obstacle_lambda
                        # print(lagrangian_value)
                        self.set_lambda(lagrangian_value)
                        if lagrangian_value > MAX_LAMBDA: # to avoid being stuck at this point, which never gets a safe trajectory
                            unsafe_plan = False
                        # print(f"unsafe_plan, set lambda to be {self.obstacle_lambda}")
                    else:
                        unsafe_plan = False
            else:
                # if self.add_lidar_error:
                #     lidar_error = self.get_lidar_error(obstacle_pos)
                #     self.set_lidar_error(lidar_error)
                global_raw_traj, info = self.plan(start=agent_pos,
                                        subgoal=self.subgoal,
                                        obstacles=obstacle_pos,
                                        robot_global_pos=robot_global_pos,
                                        robot_mat=robot_mat,
                                        lagrangian_test=self.lagrangian_test)
            global_traj = utils.get_trajectory(global_raw_traj, self.traj_len)
            self.lower_level_agent.update_traj(global_traj)
            self.lower_level_agent.passed_subgoal_index = 0 # each step, the trajectory might be totally different
            # not suitable to use the previous update rule
        action = self.lower_level_agent.step(obs)
        # print(time.time() - start_time)

        self.step_counter += 1
        return action
    def update_learnable_params(self, **kwargs):
        "Update the learnable parameters to the motion planner"
        if "Push" in self.env_name:
            self.subgoal = kwargs['subgoal']
    def set_lambda(self, obstacle_lambda):
        obstacle_lambda = min(obstacle_lambda, MAX_LAMBDA)
        super().__init__(traj_len=self.traj_len,
                      optimization_steps=self.optimization_steps,
                      total_time=self.total_time,
                      env_name=self.env_name,
                      max_distance=self.max_distance,
                      obstacle_cost_weight=self.obstacle_cost_weight,
                      obstacle_lambda=obstacle_lambda,
                    #   lidar_error=self.lidar_error
                      )
        self.motion_planner.to(device=self.device, dtype=torch.float)
    # def set_lidar_error(self, lidar_error):
    #     super().__init__(traj_len=self.traj_len,
    #                   optimization_steps=self.optimization_steps,
    #                   total_time=self.total_time,
    #                   env_name=self.env_name,
    #                   max_distance=self.max_distance,
    #                   obstacle_cost_weight=self.obstacle_cost_weight,
    #                   obstacle_lambda=self.obstacle_lambda,
    #                   lidar_error=lidar_error)
    #     self.motion_planner.to(device=self.device, dtype=torch.float)
    # def get_lidar_error(self, obstacle_pos):
    #     "obstacles are in the order of hazards, vases, pillars"
    #     bin_angle = 2 * np.pi / LIDAR_NUM
    #     dist = np.linalg.norm(obstacle_pos, axis=1)
    #     lidar_error = np.sin(bin_angle / 4) * dist * 2
    #     return lidar_error


class LowerLevelAgent():
    """The naive agent that simply follows the subgoal 
    by using the action as the vector pointing the subgoal
    It takes in the trajectory and follows the trajecotry"""
    """The trajectory for the subgoal can only be applied into the global frame
    otherwise we won't be able to track the trajectory we are tracing"""
    def __init__(self, agent_name, policy=None, subgoal_thres=0.5) -> None:
        "the trajectory here is in a global coordinate"
        self.traj = None
        self.subgoal_thres = subgoal_thres
        self.passed_subgoal_index = 0
        self.policy = policy
        if agent_name == "point" or agent_name == "mass":
            self.key_list = POINTKEYLIST
        elif agent_name == "car":
            self.key_list = CARKEYLIST
        elif agent_name == "doggo":
            self.key_list = DOGGOKEYLIST
        elif agent_name == 'ant':
            self.key_list = ANTKEYLIST
    def step(self, obs):
        assert self.traj is not None
        agent_pos = obs['robot_global_pos']
        # print(f"agent pos in lower level following module: {agent_pos / 4}")
        for i in range(self.passed_subgoal_index, self.traj.shape[2]):
            distance = np.linalg.norm((self.traj[0,:2, i] - agent_pos[:2]))
            if distance > self.subgoal_thres: # if the current waypoint is far away enough, use it as a subgoal
                if self.policy is None:
                    # we are using mass agent, which is using a trivial lower level policy
                    action = self.traj[0,:2, i] - agent_pos[:2]
                    action = utils.ego_v_xy(obs['robot_global_pos'], obs['robot_mat'], action)
                    action = action * 8
                else:
                    subgoal = self.traj[0, :2, i]
                    # change the gloabl coordinate into ego coordinate, sinve the lower level policy is trained with ego coordinate
                    subgoal = utils.ego_xy(obs['robot_global_pos'], obs['robot_mat'], subgoal)
                    lower_policy_obs = self.subgoal_to_input(subgoal, obs)
                    "for the lower lebel policy, it takes the egocentric goal as the direct input"
                    action = self.policy.select_action(lower_policy_obs, evaluate=True)
                self.passed_subgoal_index = max(i-1, 0)
                break
            elif i == (self.traj.shape[2] - 1): # reach the last one but still doesn't find the subgoal
                self.passed_subgoal_index = i - 1
                if self.policy is None:
                    action = self.traj[0, :2, -1] - agent_pos[:2]
                    action = utils.ego_v_xy(obs['robot_global_pos'], obs['robot_mat'], action)
                    action = action * 3
                else:
                    subgoal = self.traj[0, :2, -1]
                    subgoal = utils.ego_xy(obs['robot_global_pos'], obs['robot_mat'], subgoal)
                    lower_policy_obs = self.subgoal_to_input(subgoal, obs)
                    action = self.policy.select_action(lower_policy_obs, evaluate=True)
        return action
    def subgoal_to_input(self, subgoal, obs):
        output = []
        output.append(subgoal)
        for key in self.key_list:
            output.append(obs[key].reshape(-1))
        return np.concatenate(output)
        # if self.passed_subgoal_index == self.traj.shape[-1] - 1:
        #     "it means it has reached the final subgoal"
        #     subgoal_index = self.passed_subgoal_index
        # else:
        #     subgoal_index = self.passed_subgoal_index + 1
        # agent_pos = obs['robot_global_pos']
        # subgoal = self.traj[0,:2, subgoal_index]
        # action = subgoal - agent_pos[:2]
        # distance = np.linalg.norm(action)
        # # print(distance, self.subgoal_thres, self.passed_subgoal_index)
        # if distance < self.subgoal_thres:
        #     self.passed_subgoal_index += 1
        # action = utils.ego_v_xy(obs['robot_global_pos'], obs['robot_mat'], action)
        # action = action * 8
        # action = action / distance
        # return action
    def update_traj(self, traj):
        self.traj = traj
        self.passed_subgoal_index -= 2