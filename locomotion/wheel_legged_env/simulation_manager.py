import torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from .env_utils import get_relative_terrain_pos

class SimulationManager:
    def __init__(self, env):
        self.env = env
        
    def update_simulation_state(self):
        """
        更新仿真状态
        负责从仿真引擎中获取和更新原始状态数据
        """
        # Update base position and orientation
        self.env.base_pos[:] = get_relative_terrain_pos(
            self.env.robot.get_pos(), self.env.terrain_height, self.env.horizontal_scale
        )
        self.env.base_quat[:] = self.env.robot.get_quat()
        
        # Calculate Euler angles
        self.env.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.env.base_quat) * self.env.inv_base_init_quat,
                self.env.base_quat
            ),
            rpy=True,
            degrees=True,
        )
        
        # Update base velocities
        inv_base_quat = inv_quat(self.env.base_quat)
        self.env.base_lin_vel[:] = transform_by_quat(
            self.env.robot.get_vel(), inv_base_quat
        )
        self.env.base_lin_acc[:] = (self.env.base_lin_vel[:] - 
                                   self.env.last_base_lin_vel[:]) / self.env.dt
        self.env.base_ang_vel[:] = transform_by_quat(
            self.env.robot.get_ang(), inv_base_quat
        )
        self.env.base_ang_acc[:] = (self.env.base_ang_vel[:] - 
                                   self.env.last_base_ang_vel[:]) / self.env.dt
        self.env.projected_gravity = transform_by_quat(
            self.env.global_gravity, inv_base_quat
        )
        
        # Update DOF states
        self.env.dof_pos[:] = self.env.robot.get_dofs_position(self.env.motors_dof_idx)
        self.env.dof_vel[:] = self.env.robot.get_dofs_velocity(self.env.motors_dof_idx)
        self.env.dof_force[:] = self.env.robot.get_dofs_force(self.env.motors_dof_idx)
        
        # Update last states
        self.env.last_base_lin_vel[:] = self.env.base_lin_vel[:]
        self.env.last_base_ang_vel[:] = self.env.base_ang_vel[:]