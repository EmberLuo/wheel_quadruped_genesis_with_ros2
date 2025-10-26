import torch
import genesis as gs
import numpy as np

class RobotPropertiesManager:
    def __init__(self, env):
        self.env = env

    def init_robot_properties(self):
        """Initialize robot properties like default positions."""
        # Default joint angles
        self.env.basic_default_dof_pos = torch.tensor(
            [self.env.env_cfg["default_joint_angles"][name]
             for name in self.env.env_cfg["joint_names"]],
            device=self.env.device,
            dtype=gs.tc_float,
        )

        # Default positions for all environments
        default_dof_pos_list = [[self.env.env_cfg["default_joint_angles"][name]
                                for name in self.env.env_cfg["joint_names"]]
                                for _ in range(self.env.num_envs)]
        self.env.default_dof_pos = torch.tensor(
            default_dof_pos_list, device=self.env.device, dtype=gs.tc_float)

        # Initial positions for all environments
        init_dof_pos_list = [[self.env.env_cfg["joint_init_angles"][name]
                             for name in self.env.env_cfg["joint_names"]]
                             for _ in range(self.env.num_envs)]
        self.env.init_dof_pos = torch.tensor(
            init_dof_pos_list, device=self.env.device, dtype=gs.tc_float)

    def init_dof_limits(self):
        """Initialize joint limits."""
        lower = []
        upper = []
        for name in self.env.env_cfg["joint_names"]:
            if name in self.env.env_cfg["dof_limit"]:
                lower.append(self.env.env_cfg["dof_limit"][name][0])
                upper.append(self.env.env_cfg["dof_limit"][name][1])
            else:
                # For joints without specified limits (e.g., continuous joints)
                lower.append(float('-inf'))
                upper.append(float('inf'))

        self.env.dof_pos_lower = torch.tensor(lower, device=self.env.device)
        self.env.dof_pos_upper = torch.tensor(upper, device=self.env.device)

    def init_force_limits(self):
        """Initialize force/torque limits."""
        lower = np.array([[-self.env.env_cfg["safe_force"][name]
                         for name in self.env.env_cfg["joint_names"]]
                         for _ in range(self.env.num_envs)])
        upper = np.array([[self.env.env_cfg["safe_force"][name]
                          for name in self.env.env_cfg["joint_names"]]
                         for _ in range(self.env.num_envs)])

        self.env.robot.set_dofs_force_range(
            lower=torch.tensor(lower, device=self.env.device, dtype=torch.float32),
            upper=torch.tensor(upper, device=self.env.device, dtype=torch.float32),
            dofs_idx_local=self.env.motors_dof_idx,
        )