import torch
from .env_utils import gs_rand_float

class CommandManager:
    def __init__(self, env):
        self.env = env

    def init_command_ranges(self):
        """Initialize command ranges for curriculum learning."""
        self.env.command_ranges = torch.zeros(
            (self.env.num_envs, self.env.num_commands, 2),
            device=self.env.device, dtype=torch.float32)

        # Linear velocity X range
        self.env.command_ranges[:, 0, 0] = self.env.command_cfg["lin_vel_x_range"][0] * \
            self.env.command_cfg["base_range"]
        self.env.command_ranges[:, 0, 1] = self.env.command_cfg["lin_vel_x_range"][1] * \
            self.env.command_cfg["base_range"]

        # Linear velocity Y range
        self.env.command_ranges[:, 1, 0] = self.env.command_cfg["lin_vel_y_range"][0] * \
            self.env.command_cfg["base_range"]
        self.env.command_ranges[:, 1, 1] = self.env.command_cfg["lin_vel_y_range"][1] * \
            self.env.command_cfg["base_range"]

        # Angular velocity range
        self.env.command_ranges[:, 2, 0] = self.env.command_cfg["ang_vel_range"][0] * \
            self.env.command_cfg["base_range"]
        self.env.command_ranges[:, 2, 1] = self.env.command_cfg["ang_vel_range"][1] * \
            self.env.command_cfg["base_range"]

        # Leg length range for individual legs
        self.env.command_ranges[:, 3, 0] = self.env.command_cfg["leg_length_range"][0]
        self.env.command_ranges[:, 3, 1] = self.env.command_cfg["leg_length_range"][1]  # FL thigh
        self.env.command_ranges[:, 4, 0] = self.env.command_cfg["leg_length_range"][0]
        self.env.command_ranges[:, 4, 1] = self.env.command_cfg["leg_length_range"][1]  # FR thigh
        self.env.command_ranges[:, 5, 0] = self.env.command_cfg["leg_length_range"][0]
        self.env.command_ranges[:, 5, 1] = self.env.command_cfg["leg_length_range"][1]  # RL thigh
        self.env.command_ranges[:, 6, 0] = self.env.command_cfg["leg_length_range"][0]
        self.env.command_ranges[:, 6, 1] = self.env.command_cfg["leg_length_range"][1]  # RR thigh

        # Error tracking for curriculum learning
        self.env.lin_vel_error = torch.zeros(
            (self.env.num_envs, 1), device=self.env.device, dtype=torch.float32)
        self.env.ang_vel_error = torch.zeros(
            (self.env.num_envs, 1), device=self.env.device, dtype=torch.float32)
        self.env.height_error = torch.zeros(
            (self.env.num_envs, 1), device=self.env.device, dtype=torch.float32)
        self.env.curriculum_lin_vel_scale = torch.zeros(
            (self.env.num_envs, 1), device=self.env.device, dtype=torch.float32)
        self.env.curriculum_ang_vel_scale = torch.zeros(
            (self.env.num_envs, 1), device=self.env.device, dtype=torch.float32)

    def resample_commands(self, envs_idx):
        """Resample command values for specified environments."""
        for idx in envs_idx:
            for command_idx in range(self.env.num_commands):
                low = self.env.command_ranges[idx, command_idx, 0]
                high = self.env.command_ranges[idx, command_idx, 1]
                self.env.commands[idx, command_idx] = gs_rand_float(
                    low, high, (1,), self.env.device)

    def set_commands(self, envs_idx, commands):
        """Set specific command values for specified environments."""
        self.env.commands[envs_idx] = torch.tensor(
            commands, device=self.env.device, dtype=torch.float32)