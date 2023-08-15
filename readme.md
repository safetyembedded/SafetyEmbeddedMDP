## Train the lower level goal following agent
We need to first train the lower level goal following agent in an obstacle free environment. The command is shown as follows:

    python3 main_lower_level_agent.py --agent=AGENT --seed=0

Please replace AGENT with the specific agent name chosen from {point, car, ant}.
Then we save the best checkpoints into a pkl file:

    python3 transform_lower_level_agent.py --load_path=PATH --agent=AGENT

Please replace PATH with the path to the path to the best checkpoint choose AGENT from {point, car, ant}.

Then we start training the high-level RL agent:

     python3 main_mpc_motion_planning_subgoal.py --env=ENV --exp_id=EXP_ID --optimization_steps=10 --seed=SEED

Please choose ENV from {Safexp-PointPush1-v0, Safexp-CarPush1-v0, Safexp-PointPush2-v0, Safexp-CarPush2-v0}, and replace EXP_ID and SEED accordingly

