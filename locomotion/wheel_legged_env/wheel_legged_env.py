"""
Wheel-Quadruped Robot Environment

This module implements the environment for training wheel-legged quadruped robots
using reinforcement learning. The environment provides observations, rewards, 
and handles robot physics simulation.
"""

import torch
import math
import numpy as np
import cv2
import genesis as gs  # type: ignore
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat
)  # type: ignore
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from .reward_functions import RewardFunctions
from .curriculum_learning import CurriculumLearning
from .domain_randomization import DomainRandomization
from .observation_manager import ObservationManager
from .robot_controller import RobotController
from .simulation_manager import SimulationManager
from .command_manager import CommandManager
from .robot_properties_manager import RobotPropertiesManager
from .scene_manager import SceneManager
from .termination_manager import TerminationManager


class WheelLeggedEnv:
    """
    Environment for wheel-legged quadruped robot locomotion training.

    This environment simulates a quadruped robot with wheels at the end of each leg,
    providing observations and rewards for reinforcement learning training.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 curriculum_cfg, domain_rand_cfg, terrain_cfg, robot_morphs="urdf",
                 show_viewer=False, num_view=1, device="cuda", train_mode=True):
        """
        Initialize the wheel-legged environment.

        Args:
            num_envs: Number of parallel environments
            env_cfg: Environment configuration
            obs_cfg: Observation configuration
            reward_cfg: Reward configuration
            command_cfg: Command configuration
            curriculum_cfg: Curriculum learning configuration
            domain_rand_cfg: Domain randomization configuration
            terrain_cfg: Terrain configuration
            robot_morphs: Robot description format ("urdf" or "mjcf")
            show_viewer: Whether to show the viewer
            num_view: Number of environments to visualize
            device: Computing device ("cuda" or "cpu")
            train_mode: Whether in training mode
        """
        # Device and mode setup
        self.device = torch.device(device)
        self.mode = train_mode

        # Configuration parameters
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_slice_obs = obs_cfg["num_slice_obs"]
        self.history_length = obs_cfg["history_length"]
        self.num_privileged_obs = None  # 这个属性必须存在
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        # Configuration dictionaries
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.curriculum_cfg = curriculum_cfg
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg

        # Terrain parameters
        self.num_respawn_points = self.terrain_cfg["num_respawn_points"]
        self.respawn_points = self.terrain_cfg["respawn_points"]

        # Simulation parameters
        self.simulate_action_latency = True
        self.dt = 0.01  # 100Hz control frequency
        self.max_episode_length = math.ceil(
            env_cfg["episode_length_s"] / self.dt)

        # Scales and noise
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise = obs_cfg["noise"]

        # Initialize managers
        self.scene_manager = SceneManager(self)
        self.robot_properties_manager = RobotPropertiesManager(self)
        self.command_manager = CommandManager(self)
        
        # Initialize scene and robot
        self.scene_manager.create_scene(show_viewer, num_view)
        self.scene_manager.add_terrain()
        self.scene_manager.add_robot(robot_morphs)
        self.scene_manager.build_scene()

        # After building scene, we have access to robot and joint indices
        # Initialize robot properties
        self.robot_properties_manager.init_robot_properties()
        self.robot_properties_manager.init_dof_limits()
        self.robot_properties_manager.init_force_limits()
        self._init_reward_functions()
        self.command_manager.init_command_ranges()
        self._init_buffers()
        self._init_terrain_data()
        self._init_symmetry_indices()

        # 初始化新模块
        self.curriculum_learning = CurriculumLearning(self)
        self.domain_randomization = DomainRandomization(self)
        self.observation_manager = ObservationManager(self)
        self.robot_controller = RobotController(self)
        self.simulation_manager = SimulationManager(self)
        self.termination_manager = TerminationManager(self)

        # Reset environment
        self.reset()

    # ============================================================================
    # INITIALIZATION METHODS
    # ============================================================================

    def _init_reward_functions(self):
        """Initialize reward functions."""
        self.reward_functions = RewardFunctions(self)
        self.episode_sums = dict()

        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _init_buffers(self):
        """Initialize observation and state buffers."""
        # Base state buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_acc = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        # Gait-related variables
        self.feet_air_time = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.feet_contact_time = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.bool)

        # Foot indices
        self.feet_indices = [self.robot.get_link(name).idx_local
                             for name in self.env_cfg["foot_link"]]

        # Observation buffers
        self.slice_obs_buf = torch.zeros(
            (self.num_envs, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        self.history_obs_buf = torch.zeros(
            (self.num_envs, self.history_length, self.num_slice_obs),
            device=self.device, dtype=gs.tc_float)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.history_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)

        # Reward and reset buffers
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.curriculum_rew_buf = torch.zeros_like(self.rew_buf)
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int)

        # Command buffers
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"],
             self.obs_scales["dof_pos"], self.obs_scales["dof_pos"], 
             self.obs_scales["dof_pos"], self.obs_scales["dof_pos"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Action and DOF buffers
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # Contact force buffer
        self.connect_force = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)

        # Extras for logging
        self.extras = dict()

        # Reset links for termination conditions
        if (self.env_cfg["termination_if_base_contact_plane_than"] & self.mode):
            self.reset_links = [self.robot.get_link(name).idx_local
                                for name in self.env_cfg["connect_plane_links"]]

        # Foot positions
        self.foot_links = [self.robot.get_link(name)
                           for name in self.env_cfg["foot_link"]]
        self.foot_indices = [self.robot.get_link(name).idx_local
                             for name in self.env_cfg["foot_link"]]
        self.foot_positions = torch.stack([self.robot.get_link(name).get_pos()
                                          for name in self.env_cfg["foot_link"]], dim=1)
        self.foot_base_positions = torch.zeros(
            (self.num_envs, len(self.env_cfg["foot_link"]), 3),
            device=self.device, dtype=gs.tc_float)

        # Initialize foot positions
        for i, foot_link in enumerate(self.foot_links):
            self.foot_positions[:, i, :] = foot_link.get_pos()

    def _init_terrain_data(self):
        """Initialize terrain-related data for domain randomization."""
        # Domain randomization parameters
        self.friction_ratio_low = self.domain_rand_cfg["friction_ratio_range"][0]
        self.friction_ratio_range = self.domain_rand_cfg["friction_ratio_range"][1] - \
            self.friction_ratio_low

        self.base_mass_low = self.domain_rand_cfg["random_base_mass_shift_range"][0]
        self.base_mass_range = self.domain_rand_cfg["random_base_mass_shift_range"][1] - \
            self.base_mass_low

        self.other_mass_low = self.domain_rand_cfg["random_other_mass_shift_range"][0]
        self.other_mass_range = self.domain_rand_cfg["random_other_mass_shift_range"][1] - \
            self.other_mass_low

        self.dof_damping_low = self.domain_rand_cfg["damping_range"][0]
        self.dof_damping_range = self.domain_rand_cfg["damping_range"][1] - \
            self.dof_damping_low

        self.dof_armature_low = self.domain_rand_cfg["dof_armature_range"][0]
        self.dof_armature_range = self.domain_rand_cfg["dof_armature_range"][1] - \
            self.dof_armature_low

        self.kp_low = self.domain_rand_cfg["random_KP"][0]
        self.kp_range = self.domain_rand_cfg["random_KP"][1] - self.kp_low

        self.kv_low = self.domain_rand_cfg["random_KV"][0]
        self.kv_range = self.domain_rand_cfg["random_KV"][1] - self.kv_low

        self.joint_angle_low = self.domain_rand_cfg["random_default_joint_angles"][0]
        self.joint_angle_range = self.domain_rand_cfg["random_default_joint_angles"][1] - \
            self.joint_angle_low

        # Terrain buffer
        self.terrain_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int)

        # Episode lengths
        self.episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)

    def _init_symmetry_indices(self):
        """Initialize symmetry indices for joints and actions."""
        # Symmetric joint pairs
        self.mirror_joints = [
            # Front left vs front right hip
            ["FL_hip_joint", "FR_hip_joint"],
            # Front left vs front right thigh
            ["FL_thigh_joint", "FR_thigh_joint"],
            # Front left vs front right calf
            ["FL_calf_joint", "FR_calf_joint"],
            # Rear left vs rear right hip
            ["RL_hip_joint", "RR_hip_joint"],
            # Rear left vs rear right thigh
            ["RL_thigh_joint", "RR_thigh_joint"],
            # Rear left vs rear right calf
            ["RL_calf_joint", "RR_calf_joint"],
        ]

        # Joint groups
        self.joint_groups = [
            ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
                "RR_hip_joint"],      # All hips
            ["FL_thigh_joint", "FR_thigh_joint",
                "RL_thigh_joint", "RR_thigh_joint"],  # All thighs
            ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint",
                "RR_calf_joint"],    # All calves
        ]

        # Create joint index mappings
        self.joint_mirror_indices = []
        for joint_pair in self.mirror_joints:
            left_idx = self.env_cfg["joint_names"].index(joint_pair[0])
            right_idx = self.env_cfg["joint_names"].index(joint_pair[1])
            self.joint_mirror_indices.append([left_idx, right_idx])

        # Create joint group index mappings
        self.joint_group_indices = []
        for joint_group in self.joint_groups:
            group_indices = [self.env_cfg["joint_names"].index(joint_name)
                             for joint_name in joint_group]
            self.joint_group_indices.append(group_indices)

    # ============================================================================
    # COMMAND METHODS
    # ============================================================================

    def _resample_commands(self, envs_idx):
        """Resample command values for specified environments."""
        self.command_manager.resample_commands(envs_idx)

    def set_commands(self, envs_idx, commands):
        """Set specific command values for specified environments."""
        self.command_manager.set_commands(envs_idx, commands)

    # ============================================================================
    # MAIN ENVIRONMENT METHODS
    # ============================================================================

    def step(self, actions):
        """
        Execute one simulation step.

        Args:
            actions: Actions to apply to the robot

        Returns:
            tuple: (observations, privileged_observations, rewards, resets, extras)
        """
        # 应用动作控制
        self.robot_controller.apply_actions(actions)
        
        # 仿真步骤
        self.scene.step()
        
        # 更新仿真状态
        self.simulation_manager.update_simulation_state()
        
        # Update buffers
        self.episode_length_buf += 1

        # Gravity factor for rewards
        self.gravity_factor = torch.clamp(
            -self.projected_gravity[:, 2], 0, 0.7) / 0.7

        # Update gait variables
        contact_force = self.robot.get_links_net_contact_force()
        feet_contact = contact_force[:,
                                     self.feet_indices, :].norm(dim=-1) > 1.0

        # Update contact and air times
        first_contact = (feet_contact > 0.0) * (self.last_contacts == 0.0)
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.feet_air_time *= ~feet_contact
        self.feet_contact_time *= feet_contact
        self.last_contacts = feet_contact

        # Apply noise if enabled
        if self.noise["use"]:
            self.base_ang_vel[:] += torch.randn_like(self.base_ang_vel) * self.noise["ang_vel"][0] + (
                torch.rand_like(self.base_ang_vel)*2-1) * self.noise["ang_vel"][1]
            self.projected_gravity += torch.randn_like(self.projected_gravity) * self.noise["gravity"][0] + (
                torch.rand_like(self.projected_gravity)*2-1) * self.noise["gravity"][1]
            self.dof_pos[:] += torch.randn_like(self.dof_pos) * self.noise["dof_pos"][0] + (
                torch.rand_like(self.dof_pos)*2-1) * self.noise["dof_pos"][1]
            self.dof_vel[:] += torch.randn_like(self.dof_vel) * self.noise["dof_vel"][0] + (
                torch.rand_like(self.dof_vel)*2-1) * self.noise["dof_vel"][1]

        # Update contact forces
        self.connect_force = self.robot.get_links_net_contact_force()

        # Update last states
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        self.episode_lengths += 1

        # Resample commands periodically
        envs_idx = (
            (self.episode_length_buf %
             int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )

        # Update terrain buffer
        self.terrain_buf[:] = 0
        self.terrain_buf = self.command_ranges[:, 0, 1] > \
            self.command_cfg["lin_vel_x_range"][1] * 0.9
        self.terrain_buf &= self.command_ranges[:, 2, 1] > \
            self.command_cfg["ang_vel_range"][1] * 0.9
        self.terrain_buf[:int(self.num_envs*0.4)] = 1  # 40% go to terrain

        # Check termination conditions
        if self.mode:
            self.termination_manager.check_termination()

        # Handle timeouts
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(
            as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # Reset environments if needed
        if self.mode:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute rewards
        if self.mode:
            self.rew_buf[:] = 0.0
            for name in self.reward_scales.keys():
                reward_func = getattr(self.reward_functions, f"_reward_{name}")
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        # Compute curriculum rewards
        self.lin_vel_error = torch.abs(
            self.commands[:, 0] - self.base_lin_vel[:, 0]).mean()
        self.ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2]).mean()
        if self.mode:
            self._resample_commands(envs_idx)
            self.curriculum_learning.update_commands()
            # 每100个step更新一次控制参数和终止条件
            if self.episode_length_buf[0] % 100 == 0:
                self.curriculum_learning.update_termination_conditions()
                self.curriculum_learning.update_control_parameters()

        # 计算观测值
        slice_obs = self.observation_manager.compute_observations()
        self.obs_buf = self.observation_manager.update_history(slice_obs)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Update statistics
        self.survive_ratio = (~self.reset_buf.bool()).float().mean()
        self.mean_lin_vel_error = self.lin_vel_error.mean()
        self.mean_ang_vel_error = self.ang_vel_error.mean()
        self.linx_range_up_threshold = self.curriculum_cfg["lin_vel_err_range"][1]
        self.angv_range_up_threshold = self.curriculum_cfg["ang_vel_err_range"][1]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """Get current observations."""
        return self.obs_buf

    def get_privileged_observations(self):
        """Get privileged observations (not implemented)."""
        return None

    # ============================================================================
    # RESET METHODS
    # ============================================================================

    def reset_idx(self, envs_idx):
        """
        Reset specified environments.

        Args:
            envs_idx: Indices of environments to reset
        """
        if len(envs_idx) == 0:
            return

        # Reset DOFs
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset base
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if self.terrain_cfg["terrain"]:
            if self.mode:
                terrain_buf = self.terrain_buf[envs_idx]
                terrain_idx = envs_idx[terrain_buf.nonzero(
                    as_tuple=False).flatten()]
                non_terrain_idx = envs_idx[(~terrain_buf).nonzero(
                    as_tuple=False).flatten()]

                # Set terrain positions
                if len(terrain_idx) > 0:
                    n = len(terrain_idx)
                    random_idx = torch.randint(
                        1, self.num_respawn_points, (n,))
                    selected_pos = self.base_terrain_pos[random_idx]
                    self.base_pos[terrain_idx] = selected_pos

                # Set non-terrain positions
                if len(non_terrain_idx) > 0:
                    self.base_pos[non_terrain_idx] = self.base_terrain_pos[0]
            else:
                self.base_pos[envs_idx] = self.base_init_pos
        else:
            self.base_pos[envs_idx] = self.base_init_pos

        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Reset gait variables
        self.feet_air_time[envs_idx] = 0.0
        self.feet_contact_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False

        # Fill extras with episode statistics
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() /
                self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

        if self.mode:
            self.domain_randomization.apply_randomization(envs_idx)

        self.episode_lengths[envs_idx] = 0.0
        self.reset_buf[envs_idx] = 0

    def reset(self):
        """Reset all environments."""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reset_buf[:] = 0
        return self.obs_buf, None

    # ============================================================================
    # TERMINATION AND DOMAIN RANDOMIZATION
    # ============================================================================

    def check_termination(self):
        """Check termination conditions."""
        self.termination_manager.check_termination()

    def domain_rand(self, envs_idx):
        """
        Apply domain randomization to specified environments.

        Args:
            envs_idx: Indices of environments to apply randomization
        """
        self.domain_randomization.apply_randomization(envs_idx)

    def curriculum_commands(self):
        """Update command ranges based on curriculum learning."""
        self.curriculum_learning.update_commands()