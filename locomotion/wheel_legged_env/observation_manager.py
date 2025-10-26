import torch
import numpy as np
from genesis.utils.geom import transform_by_quat, inv_quat


class ObservationManager:
    def __init__(self, env):
        self.env = env

    def compute_observations(self):
        """
        计算当前观测值
        负责将原始状态数据处理成强化学习算法可用的观测值
        具体维度分解如下：
        ang_vel (角速度): 3维
        base_ang_vel[:, 0] - 绕X轴旋转角速度
        base_ang_vel[:, 1] - 绕Y轴旋转角速度
        base_ang_vel[:, 2] - 绕Z轴旋转角速度
        gravity (重力投影): 3维
        projected_gravity[:, 0] - X轴重力分量
        projected_gravity[:, 1] - Y轴重力分量
        projected_gravity[:, 2] - Z轴重力分量
        rel_dof_pos (相对关节位置): 12维
        机器人有16个自由度(4条腿*4个关节)
        其中12个是普通关节(hip, thigh, calf)，4个是轮子关节(foot)
        rel_dof_pos只包含12个普通关节相对于默认位置的偏差
        self.env.joint_dof_idx_np索引标识了这12个关节
        dof_vel (关节速度): 16维
        所有16个关节的速度值
        包括4个普通关节和4个轮子关节
        actions (当前动作): 16维
        当前应用到所有16个关节的动作值
        总计：3 + 3 + 12 + 16 + 16 = 50维
        """
        # Base angular velocity
        ang_vel = self.env.base_ang_vel * self.env.obs_scales["ang_vel"]

        # Projected gravity
        gravity = self.env.projected_gravity

        # Relative joint positions
        rel_dof_pos = (self.env.dof_pos[:, self.env.joint_dof_idx_np] -
                       self.env.default_dof_pos[:, self.env.joint_dof_idx_np]) * \
            self.env.obs_scales["dof_pos"]

        # Joint velocities
        dof_vel = self.env.dof_vel * self.env.obs_scales["dof_vel"]

        # Current actions
        actions = self.env.actions

        slice_obs = torch.cat([
            ang_vel, gravity, rel_dof_pos, dof_vel, actions
        ], axis=-1)

        return slice_obs

    def update_history(self, slice_obs):
        """更新历史观测值"""
        if self.env.history_length > 1:
            self.env.history_obs_buf[:, :-1,
                                     :] = self.env.history_obs_buf[:, 1:, :].clone()
        self.env.history_obs_buf[:, -1, :] = slice_obs

        obs_buf = torch.cat([self.env.history_obs_buf, slice_obs.unsqueeze(1)],
                            dim=1).view(self.env.num_envs, -1)
        obs_buf = torch.cat(
            [obs_buf, self.env.commands * self.env.commands_scale], axis=-1)

        return obs_buf
