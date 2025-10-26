import torch
from .env_utils import adjust_scale

class CurriculumLearning:
    def __init__(self, env):
        self.env = env
        
    def update_commands(self):
        """更新课程学习命令范围"""
        # Update errors
        self.env.lin_vel_error /= self.env.num_envs
        self.env.ang_vel_error /= self.env.num_envs
        
        # Adjust linear velocity range
        lin_min_range, lin_max_range = adjust_scale(
            self.env.lin_vel_error,
            self.env.curriculum_cfg["lin_vel_err_range"][0],
            self.env.curriculum_cfg["lin_vel_err_range"][1],
            self.env.curriculum_lin_vel_scale,
            self.env.curriculum_cfg["curriculum_lin_vel_step"],
            self.env.curriculum_cfg["curriculum_lin_vel_min_range"],
            self.env.command_cfg["lin_vel_x_range"]
        )
        self.env.command_ranges[:, 0, 0] = lin_min_range.squeeze()
        self.env.command_ranges[:, 0, 1] = lin_max_range.squeeze()
        
        # Adjust angular velocity range
        ang_min_range, ang_max_range = adjust_scale(
            self.env.ang_vel_error,
            self.env.curriculum_cfg["ang_vel_err_range"][0],
            self.env.curriculum_cfg["ang_vel_err_range"][1],
            self.env.curriculum_ang_vel_scale,
            self.env.curriculum_cfg["curriculum_ang_vel_step"],
            self.env.curriculum_cfg["curriculum_ang_vel_min_range"],
            self.env.command_cfg["ang_vel_range"]
        )
        self.env.command_ranges[:, 2, 0] = ang_min_range.squeeze()
        self.env.command_ranges[:, 2, 1] = ang_max_range.squeeze()
        
    def update_termination_conditions(self):
        """根据学习进度调整终止条件"""
        # 根据平均存活时间调整终止条件的严格程度
        mean_survival_time = self.env.episode_lengths.mean()
        target_survival_time = self.env.env_cfg["episode_length_s"] / self.env.dt * 0.5  # 目标为episode长度的一半
        
        # 如果平均存活时间太短，放宽终止条件
        if mean_survival_time < target_survival_time * 0.5:
            # 逐步放宽角度限制
            current_roll_limit = self.env.env_cfg["termination_if_roll_greater_than"]
            current_pitch_limit = self.env.env_cfg["termination_if_pitch_greater_than"]
            
            # 增加5度的宽容度，但不超过最大值
            new_roll_limit = min(current_roll_limit + 1.0, 90.0)
            new_pitch_limit = min(current_pitch_limit + 1.0, 60.0)
            
            self.env.env_cfg["termination_if_roll_greater_than"] = new_roll_limit
            self.env.env_cfg["termination_if_pitch_greater_than"] = new_pitch_limit
            
    def update_control_parameters(self):
        """根据学习进度调整控制参数"""
        # 根据平均存活时间调整PD控制器参数
        mean_survival_time = self.env.episode_lengths.mean()
        target_survival_time = self.env.env_cfg["episode_length_s"] / self.env.dt * 0.5
        
        # 如果存活时间足够长，可以增加控制器的激进程度
        if mean_survival_time > target_survival_time * 1.5:
            # 增加PD增益以获得更精确的控制
            current_kp = self.env.env_cfg["joint_kp"]
            current_kv = self.env.env_cfg["joint_kv"]
            
            # 适度增加控制增益
            new_kp = min(current_kp * 1.05, 80.0)
            new_kv = min(current_kv * 1.05, 2.0)
            
            self.env.env_cfg["joint_kp"] = new_kp
            self.env.env_cfg["joint_kv"] = new_kv
            
            # 更新实际的控制器参数
            self.env.robot.set_dofs_kp(
                [new_kp] * self.env.num_actions, 
                self.env.motors_dof_idx
            )
            self.env.robot.set_dofs_kv(
                [new_kv] * self.env.num_actions, 
                self.env.motors_dof_idx
            )