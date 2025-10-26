import torch
import numpy as np

class RobotController:
    def __init__(self, env):
        self.env = env
        
    def apply_actions(self, actions):
        """应用动作到机器人"""
        # Clip actions
        self.env.actions = torch.clip(
            actions, -self.env.env_cfg["clip_actions"], self.env.env_cfg["clip_actions"]
        )
        
        # Apply action latency if enabled
        exec_actions = self.env.last_actions if self.env.simulate_action_latency else self.env.actions
        
        # Apply position control to joints
        target_dof_pos = (exec_actions[:, self.env.joint_dof_idx_np] * 
                         self.env.env_cfg["joint_action_scale"] + 
                         self.env.default_dof_pos[:, self.env.joint_dof_idx_np])
        
        target_dof_pos = torch.clamp(
            target_dof_pos,
            min=self.env.dof_pos_lower[self.env.joint_dof_idx_np],
            max=self.env.dof_pos_upper[self.env.joint_dof_idx_np]
        )
        self.env.robot.control_dofs_position(target_dof_pos, self.env.joint_dof_idx)
        
        # Apply velocity control to wheels
        target_dof_vel = (exec_actions[:, self.env.wheel_dof_idx_np] * 
                         self.env.env_cfg["wheel_action_scale"])
        self.env.robot.control_dofs_velocity(target_dof_vel, self.env.wheel_dof_idx)