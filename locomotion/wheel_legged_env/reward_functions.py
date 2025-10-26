"""
Reward Functions for Wheel-Legged Quadruped Robot

This module implements various reward functions for training locomotion policies.
Rewards are categorized into different groups based on their functionality:
- Velocity Tracking: Rewards for following velocity commands
- Posture Control: Rewards for maintaining desired body posture
- Action Regularization: Penalties for undesirable action patterns
- Gait Quality: Rewards for proper gait patterns and foot placement
- Symmetry: Rewards for symmetric leg movements
- Constraints: Penalties for violating physical constraints
"""

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat


class RewardFunctions:
    """
    A collection of reward functions for wheel-legged quadruped robot locomotion.

    Each method prefixed with '_reward_' implements a specific reward component
    that contributes to the overall reward signal for reinforcement learning.
    """

    def __init__(self, env):
        """
        Initialize reward functions with environment reference.

        Args:
            env: The wheel-legged environment instance
        """
        self.env = env

    # ============================================================================
    # VELOCITY TRACKING REWARDS
    # ============================================================================

    def _reward_tracking_lin_x_vel(self):
        """
        Reward tracking of linear velocity commands (x axis) using exponential kernel.

        Encourages the robot to match the commanded forward/backward velocity.
        """
        lin_vel_error = torch.square(
            self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        reward = torch.exp(-lin_vel_error /
                           (self.env.reward_cfg["tracking_linx_sigma"]**2))
        return reward * self.env.gravity_factor

    def _reward_tracking_lin_y_vel(self):
        """
        Reward tracking of linear velocity commands (y axis) using exponential kernel.

        Encourages the robot to match the commanded lateral velocity.
        """
        lin_vel_error = torch.square(
            self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        reward = torch.exp(-lin_vel_error /
                           (self.env.reward_cfg["tracking_liny_sigma"]**2))
        return reward * self.env.gravity_factor

    def _reward_tracking_ang_vel(self):
        """
        Reward tracking of angular velocity commands (yaw) using exponential kernel.

        Encourages the robot to match the commanded yaw rotation velocity.
        """
        ang_vel_error = torch.square(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error /
                           (self.env.reward_cfg["tracking_ang_sigma"]**2))
        return reward * self.env.gravity_factor

    def _reward_tracking_lin_xy_yaw_frame(self):
        """
        Reward tracking of linear velocity commands in gravity-aligned robot frame.

        Transforms velocity to yaw-aligned frame before computing reward for better
        heading-invariant velocity tracking.
        """
        # Compute inverse yaw quaternion
        yaw_quat_inv = torch.zeros_like(self.env.base_quat)
        yaw_quat_inv[:, 0] = self.env.base_quat[:, 0]   # w
        yaw_quat_inv[:, 1] = -self.env.base_quat[:, 1]  # x
        yaw_quat_inv[:, 2] = -self.env.base_quat[:, 2]  # y
        yaw_quat_inv[:, 3] = self.env.base_quat[:, 3]   # z

        # Transform world velocity to yaw-aligned robot frame
        vel_yaw = transform_by_quat(
            self.env.robot.get_vel()[:, :3], yaw_quat_inv)

        # Compute error in xy plane
        lin_vel_error = torch.sum(
            torch.square(self.env.commands[:, :2] - vel_yaw[:, :2]), dim=1)

        # Standard deviation of 0.1
        reward = torch.exp(-lin_vel_error / (0.1**2))
        return reward * self.env.gravity_factor

    # ============================================================================
    # POSTURE CONTROL REWARDS
    # ============================================================================

    def _reward_tracking_leg_length(self):
        """
        Reward for tracking desired leg length/position.

        Tracks thigh joint positions as proxy for leg configuration.
        """
        # Thigh joint tracking for leg configuration
        thigh_error = torch.square(
            self.env.dof_pos[:, 1] - self.env.commands[:, 3])  # FL thigh
        thigh_error += torch.square(
            self.env.dof_pos[:, 4] - self.env.commands[:, 4])  # FR thigh
        thigh_error += torch.square(
            self.env.dof_pos[:, 7] - self.env.commands[:, 5])  # RL thigh
        thigh_error += torch.square(
            self.env.dof_pos[:, 10] - self.env.commands[:, 6])  # RR thigh
        
        # Return reward instead of error (negative error)
        reward = torch.exp(-thigh_error / 0.1)  # Using fixed sigma value
        return reward * self.env.gravity_factor

    def _reward_projected_gravity(self):
        """
        Reward for maintaining body horizontal orientation.

        Penalizes deviation of projected gravity vector from [0, 0, -1].
        """
        reward = torch.sum(torch.square(
            self.env.projected_gravity[:, :2]), dim=1)
        return reward * self.env.gravity_factor

    def _reward_similar_legged(self):
        """
        Reward for symmetric leg configurations.

        Encourages similar configurations between front-left/front-right legs
        and rear-left/rear-right legs to suppress leg splaying.
        """
        # Front legs symmetry (FL vs FR)
        front_leg_error = torch.sum(
            torch.square(self.env.dof_pos[:, 0:4] - self.env.dof_pos[:, 4:8]), dim=1)
        
        # Rear legs symmetry (RL vs RR)
        rear_leg_error = torch.sum(
            torch.square(self.env.dof_pos[:, 8:12] - self.env.dof_pos[:, 12:16]), dim=1)
        
        # Total symmetry error
        legged_error = front_leg_error + rear_leg_error
        
        reward = torch.exp(-legged_error /
                           self.env.reward_cfg["tracking_similar_legged_sigma"])
        return reward * self.env.gravity_factor

    # ============================================================================
    # ACTION REGULARIZATION REWARDS
    # ============================================================================

    def _reward_lin_vel_z(self):
        """
        Penalize vertical base linear velocity.

        Discourages unwanted vertical movement of the robot body.
        """
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_joint_action_rate(self):
        """
        Penalize rapid changes in joint actions.

        Encourages smooth joint movements by penalizing action differences.
        """
        return torch.sum(torch.square(
            self.env.last_actions[:, self.env.joint_dof_idx_np] -
            self.env.actions[:, self.env.joint_dof_idx_np]), dim=1)

    def _reward_wheel_action_rate(self):
        """
        Penalize rapid changes in wheel actions.

        Encourages smooth wheel velocity control.
        """
        return torch.sum(torch.square(
            self.env.last_actions[:, self.env.wheel_dof_idx_np] -
            self.env.actions[:, self.env.wheel_dof_idx_np]), dim=1)

    def _reward_dof_acc(self):
        """
        Penalize high joint accelerations.

        Reduces mechanical stress by discouraging rapid velocity changes.
        """
        return torch.sum(torch.square(
            (self.env.dof_vel - self.env.last_dof_vel) / self.env.dt), dim=1)

    def _reward_dof_force(self):
        """
        Penalize high joint forces/torques.

        Encourages energy-efficient movements with lower actuation forces.
        """
        return torch.sum(torch.square(self.env.dof_force), dim=1)

    def _reward_ang_vel_xy(self):
        """
        Penalize base roll and pitch angular velocities.

        Discourages unwanted rotational movements around horizontal axes.
        """
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    # ============================================================================
    # GAIT QUALITY REWARDS
    # ============================================================================

    def _reward_feet_distance_xy_exp(self):
        """
        Reward proper foot placement in xy plane.

        Encourages feet to be positioned at desired locations relative to body.
        """
        # Desired stance parameters
        stance_width = 0.4   # Lateral distance between feet
        stance_length = 0.4  # Longitudinal distance between feet
        std = 0.1            # Standard deviation for exponential kernel

        # Get foot positions in world frame
        feet_pos_world = torch.stack([
            self.env.robot.get_link(name).get_pos()
            for name in self.env.env_cfg["foot_link"]], dim=1)

        # Convert to body frame
        base_pos_world = self.env.base_pos.unsqueeze(1)
        cur_footsteps_translated = feet_pos_world - base_pos_world

        inv_base_quat = inv_quat(self.env.base_quat)
        footsteps_in_body_frame = torch.zeros(
            self.env.num_envs, 4, 3, device=self.env.device)

        for i in range(4):
            footsteps_in_body_frame[:, i, :] = transform_by_quat(
                cur_footsteps_translated[:, i, :], inv_base_quat)

        # Compute desired foot positions
        stance_width_tensor = stance_width * \
            torch.ones([self.env.num_envs, 1], device=self.env.device)
        stance_length_tensor = stance_length * \
            torch.ones([self.env.num_envs, 1], device=self.env.device)

        # Desired X positions: [length/2, length/2, -length/2, -length/2]
        # Corresponds to: FL, FR, RL, RR
        desired_xs = torch.cat([
            stance_length_tensor / 2, stance_length_tensor / 2,
            -stance_length_tensor / 2, -stance_length_tensor / 2], dim=1)

        # Desired Y positions: [width/2, -width/2, width/2, -width/2]
        # Corresponds to: FL, FR, RL, RR
        desired_ys = torch.cat([
            stance_width_tensor / 2, -stance_width_tensor / 2,
            stance_width_tensor / 2, -stance_width_tensor / 2], dim=1)

        # Compute position errors
        stance_diff_x = torch.square(
            desired_xs - footsteps_in_body_frame[:, :, 0])
        stance_diff_y = torch.square(
            desired_ys - footsteps_in_body_frame[:, :, 1])

        # Exponential reward
        stance_diff = stance_diff_x + stance_diff_y
        reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))

        return reward * self.env.gravity_factor

    def _reward_feet_air_time(self):
        """
        Reward appropriate foot air and contact times.

        Encourages proper gait timing with optimal stance and swing phases.
        """
        # Parameters
        target_contact_time = 0.25   # Target contact time
        target_air_time = 0.25       # Target air time
        velocity_threshold = 0.1     # Minimum velocity for gait rewards

        # Calculate how close we are to target times
        contact_time_error = torch.square(
            self.env.feet_contact_time - target_contact_time)
        air_time_error = torch.square(
            self.env.feet_air_time - target_air_time)
        
        # Combine errors for all feet
        timing_error = torch.sum(contact_time_error + air_time_error, dim=1)
        
        # Exponential reward (lower error = higher reward)
        reward = torch.exp(-timing_error / 0.1)  # Using fixed sigma value
        
        # Only apply when moving significantly
        moving = torch.linalg.norm(self.env.commands[:, :2], dim=1) > velocity_threshold
        reward = torch.where(moving, reward, torch.ones_like(reward))
        
        return reward * self.env.gravity_factor
    
    def _reward_feet_contact(self):
        """
        Reward appropriate number of foot contacts.

        Encourages stable stance with desired number of feet in contact.
        """
        expect_contact_num = 2  # Desired number of contacts
        contact = self.env.last_contacts
        contact_num = torch.sum(contact, dim=1)

        # Penalize deviation from expected contact number
        reward = (contact_num != expect_contact_num).float()

        # Only apply when moving
        reward *= torch.linalg.norm(self.env.commands[:, :2], dim=1) > 0.1
        return reward * self.env.gravity_factor

    def _reward_feet_stumble(self):
        """
        Penalize feet hitting vertical surfaces.

        Discourages foot collisions with obstacles or uneven terrain.
        """
        # Get contact forces
        contact_force = self.env.robot.get_links_net_contact_force()
        forces = contact_force[:, self.env.feet_indices, :]

        # Separate vertical and horizontal forces
        forces_z = torch.abs(forces[:, :, 2])
        forces_xy = torch.linalg.norm(forces[:, :, :2], dim=2)

        # Penalize when horizontal forces dominate (indicating side impacts)
        reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
        return reward * self.env.gravity_factor

    # ============================================================================
    # SYMMETRY REWARDS
    # ============================================================================

    def _reward_joint_mirror(self):
        """
        Reward mirror symmetry in joint positions.

        Encourages symmetric leg movements for balanced locomotion.
        """
        reward = torch.zeros(self.env.num_envs, device=self.env.device)

        # Compute differences between symmetric joints
        for joint_pair_indices in self.env.joint_mirror_indices:
            left_idx, right_idx = joint_pair_indices
            diff = torch.square(
                self.env.dof_pos[:, left_idx] - self.env.dof_pos[:, right_idx])
            reward += diff

        # Normalize by number of joint pairs
        reward *= 1 / \
            len(self.env.mirror_joints) if len(
                self.env.mirror_joints) > 0 else 0
        return reward * self.env.gravity_factor

    def _reward_action_mirror(self):
        """
        Reward mirror symmetry in actions.

        Encourages symmetric control actions for balanced locomotion.
        """
        reward = torch.zeros(self.env.num_envs, device=self.env.device)

        # Compute differences between symmetric actions
        for joint_pair_indices in self.env.joint_mirror_indices:
            left_idx, right_idx = joint_pair_indices
            diff = torch.square(
                torch.abs(self.env.actions[:, left_idx]) -
                torch.abs(self.env.actions[:, right_idx]))
            reward += diff

        # Normalize by number of joint pairs
        reward *= 1 / \
            len(self.env.mirror_joints) if len(
                self.env.mirror_joints) > 0 else 0
        return reward * self.env.gravity_factor

    def _reward_action_sync(self):
        """
        Reward synchronized actions within joint groups.

        Encourages coordinated movements within functional groups (e.g., all hips).
        """
        reward = torch.zeros(self.env.num_envs, device=self.env.device)

        # Compute synchronization within each joint group
        for group_indices in self.env.joint_group_indices:
            if len(group_indices) < 2:
                continue  # Need at least 2 joints for comparison

            # Get absolute actions for this group
            actions = torch.abs(self.env.actions[:, group_indices])

            # Compute mean action for the group
            mean_actions = torch.mean(actions, dim=1, keepdim=True)

            # Compute variance from mean (we want to minimize this)
            variance = torch.mean(torch.square(actions - mean_actions), dim=1)
            reward += variance

        # Normalize by number of groups
        reward *= 1 / \
            len(self.env.joint_groups) if len(self.env.joint_groups) > 0 else 0
        return reward * self.env.gravity_factor

    # ============================================================================
    # CONSTRAINT REWARDS
    # ============================================================================

    def _reward_collision(self):
        """
        Penalize base collisions with ground.

        Discourages the robot body from touching the ground.
        """
        collision = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=gs.tc_float)
        
        if self.env.env_cfg["termination_if_base_contact_plane_than"]:
            for idx in self.env.reset_links:
                collision += torch.square(
                    self.env.connect_force[:, idx, :]).sum(dim=1)
        
        return collision * self.env.gravity_factor

    # ============================================================================
    # SURVIVAL REWARDS
    # ============================================================================

    def _reward_survive(self):
        """
        Reward for staying alive (not terminated).

        Provides a baseline reward for each step the robot remains active.
        """
        return torch.ones(
            self.env.num_envs, dtype=torch.float, device=self.env.device, requires_grad=False)
