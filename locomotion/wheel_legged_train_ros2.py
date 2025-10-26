import argparse
import os
import pickle
import shutil
import torch

from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# 添加ROS2导入（如果在ROS2环境中）
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 16,
        "urdf": "assets/go2w_description/urdf/go2w_description.urdf",

        # joint names
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": -0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": -0.0,
            "FL_thigh_joint": 0.7,
            "FR_thigh_joint": 0.7,
            "RL_thigh_joint": 0.7,
            "RR_thigh_joint": 0.7,
            "FL_calf_joint": -1.4,
            "FR_calf_joint": -1.4,
            "RL_calf_joint": -1.4,
            "RR_calf_joint": -1.4,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        },
        "joint_init_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": -0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": -0.0,
            "FL_thigh_joint": 0.7,
            "FR_thigh_joint": 0.7,
            "RL_thigh_joint": 0.7,
            "RR_thigh_joint": 0.7,
            "FL_calf_joint": -1.4,
            "FR_calf_joint": -1.4,
            "RL_calf_joint": -1.4,
            "RR_calf_joint": -1.4,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        },
        "joint_names": [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FL_foot_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FR_foot_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RL_foot_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RR_foot_joint",
        ],
        "joint_type": {
            "FL_hip_joint": "joint",
            "FL_thigh_joint": "joint",
            "FL_calf_joint": "joint",
            "FL_foot_joint": "wheel",
            "FR_hip_joint": "joint",
            "FR_thigh_joint": "joint",
            "FR_calf_joint": "joint",
            "FR_foot_joint": "wheel",
            "RL_hip_joint": "joint",
            "RL_thigh_joint": "joint",
            "RL_calf_joint": "joint",
            "RL_foot_joint": "wheel",
            "RR_hip_joint": "joint",
            "RR_thigh_joint": "joint",
            "RR_calf_joint": "joint",
            "RR_foot_joint": "wheel",
        },
        "dof_limit": {
            "FL_hip_joint": [-1.0472, 1.0472],
            "FL_thigh_joint": [-1.5708, 3.4907],
            "FL_calf_joint": [-2.7227, -0.83776],
            "FR_hip_joint": [-1.0472, 1.0472],
            "FR_thigh_joint": [-1.5708, 3.4907],
            "FR_calf_joint": [-2.7227, -0.83776],
            "RL_hip_joint": [-1.0472, 1.0472],
            "RL_thigh_joint": [-0.5236, 4.5379],
            "RL_calf_joint": [-2.7227, -0.83776],
            "RR_hip_joint": [-1.0472, 1.0472],
            "RR_thigh_joint": [-0.5236, 4.5379],
            "RR_calf_joint": [-2.7227, -0.83776],
        },
        "safe_force": {
            "FL_hip_joint": 23.7,
            "FL_thigh_joint": 23.7,
            "FL_calf_joint": 35.55,
            "FL_foot_joint": 23.7,
            "FR_hip_joint": 23.7,
            "FR_thigh_joint": 23.7,
            "FR_calf_joint": 35.55,
            "FR_foot_joint": 23.7,
            "RL_hip_joint": 23.7,
            "RL_thigh_joint": 23.7,
            "RL_calf_joint": 35.55,
            "RL_foot_joint": 23.7,
            "RR_hip_joint": 23.7,
            "RR_thigh_joint": 23.7,
            "RR_calf_joint": 35.55,
            "RR_foot_joint": 23.7,
        },
        "joint_kp": 30.0,
        "joint_kv": 0.8,
        "wheel_kv": 1.0,
        "damping": 0.01,
        "armature": 0.002,
        "termination_if_roll_greater_than": 60,
        "termination_if_pitch_greater_than": 35,
        "termination_if_base_contact_plane_than": True,
        "connect_plane_links": [
            "base",
            "FL_calf",
            "FL_thigh",
            "FR_calf",
            "FR_thigh",
            "RL_calf",
            "RL_thigh",
            "RR_calf",
            "RR_thigh",
        ],
        "foot_link": [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ],
        "base_init_pos": {
            "urdf": [0.0, 0.0, 0.45],
            "mjcf": [0.0, 0.0, 0.45],
        },
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 3.0,
        "joint_action_scale": 0.5,
        "wheel_action_scale": 10.0,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "convexify": True,
        "decimate_aggressiveness": 4,
    }
    obs_cfg = {
        "num_obs": 507,
        "num_slice_obs": 50,
        "history_length": 9,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "noise": {
            "use": True,
            "ang_vel": [0.01, 0.01],
            "dof_pos": [0.01, 0.01],
            "dof_vel": [0.01, 0.01],
            "gravity": [0.01, 0.01],
        }
    }
    reward_cfg = {
        "tracking_linx_sigma": 0.2,
        "tracking_liny_sigma": 0.01,
        "tracking_ang_sigma": 0.6,
        "tracking_height_sigma": 0.005,
        "tracking_similar_legged_sigma": 0.01,
        "feet_distance": [0.3, 0.8],
        "reward_scales": {
            "survive": 2.0,
            "tracking_lin_x_vel": 5.0,
            "tracking_lin_y_vel": 0.0,
            "tracking_ang_vel": 1.5,
            "tracking_lin_xy_yaw_frame": 1.0,
            "tracking_leg_length": -5.0,
            "lin_vel_z": -0.02,
            "projected_gravity": -12.0,
            "feet_distance_xy_exp": 1.0,
            "feet_air_time": 1.0,
            "feet_contact": -1.0,
            "feet_stumble": -0.7,
            "joint_mirror": -1.0,
            "action_mirror": -0.6,
            "action_sync": -0.6,
            "joint_action_rate": -0.02,
            "wheel_action_rate": -0.01,
            "ang_vel_xy": -0.02,
            "dof_acc": -1e-7,
            "dof_force": -1e-6,
            "collision": -0.00015,
            "similar_legged": 0.0,
        },
    }
    command_cfg = {
        "num_commands": 7,
        "base_range": 0.8,
        "lin_vel_x_range": [-1.2, 1.2],
        "lin_vel_y_range": [-1.2, 1.2],
        "ang_vel_range": [-6.28, 6.28],
        "leg_length_range": [0.0, 1.0],
    }
    curriculum_cfg = {
        "curriculum_lin_vel_step": 0.015,
        "curriculum_ang_vel_step": 0.00015,
        "curriculum_lin_vel_min_range": 0.3,
        "curriculum_ang_vel_min_range": 0.1,
        "lin_vel_err_range": [0.25, 0.45],
        "ang_vel_err_range": [0.25, 0.45],
        "damping_descent": False,
        "dof_damping_descent": [0.2, 0.005, 0.001, 0.4],
    }
    domain_rand_cfg = {
        "friction_ratio_range": [0.2, 1.6],
        "random_base_mass_shift_range": [-1.5, 1.5],
        "random_other_mass_shift_range": [-0.1, 0.1],
        "random_base_com_shift": 0.05,
        "random_other_com_shift": 0.01,
        "random_KP": [0.8, 1.2],
        "random_KV": [0.8, 1.2],
        "random_default_joint_angles": [-0.03, 0.03],
        "damping_range": [0.8, 1.2],
        "dof_stiffness_range": [0.0, 0.0],
        "dof_armature_range": [0.0, 0.008],
    }
    terrain_cfg = {
        "terrain": True,
        "train": "agent_train_gym",
        "eval": "agent_eval_gym",
        "num_respawn_points": 3,
        "respawn_points": [
            [-5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.08],
        ],
        "horizontal_scale": 0.1,
        "vertical_scale": 0.001,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg


def train_with_ros2(exp_name="wheel-quadruped-walking", num_envs=1024, max_iterations=7000):
    """支持ROS2的训练函数"""
    gs.init(logging_level="warning", backend=gs.gpu)

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name, max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg,
        show_viewer=False,
        num_view=10,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg,
            domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=max_iterations,
                 init_at_random_ep_len=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str,
                        default="wheel-quadruped-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=7000)
    parser.add_argument("--use_ros2", action="store_true", 
                        help="Use ROS2 for communication")
    args = parser.parse_args()

    if args.use_ros2 and ROS2_AVAILABLE:
        # 在ROS2节点中运行训练
        rclpy.init()
        node = Node('wheel_quadruped_trainer')
        node.get_logger().info('Starting training with ROS2 support')
        
        # 这里需要进一步实现ROS2集成
        # 可以创建一个训练管理器类来处理训练过程
        
        rclpy.shutdown()
    else:
        # 直接运行训练
        gs.init(logging_level="warning", backend=gs.gpu)
        
        log_dir = f"logs/{args.exp_name}"
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        env = WheelLeggedEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
            command_cfg=command_cfg, curriculum_cfg=curriculum_cfg,
            domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg,
            show_viewer=False,
            num_view=10,
        )

        runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg,
                domain_rand_cfg, terrain_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

        runner.learn(num_learning_iterations=args.max_iterations,
                     init_at_random_ep_len=True)


if __name__ == "__main__":
    main()