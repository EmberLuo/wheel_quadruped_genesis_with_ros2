# domain_randomization.py
import torch
import numpy as np

class DomainRandomization:
    def __init__(self, env):
        self.env = env
        
    def apply_randomization(self, envs_idx):
        """应用域随机化"""
        if len(envs_idx) == 0:
            return
            
        # Friction randomization
        friction_ratio = (self.env.friction_ratio_low + self.env.friction_ratio_range * 
                         torch.rand(len(envs_idx), self.env.robot.n_links, device=self.env.device))
        self.env.robot.set_friction_ratio(
            friction_ratio=friction_ratio,
            links_idx_local=np.arange(0, self.env.robot.n_links),
            envs_idx=envs_idx
        )
        
        # Mass randomization
        base_mass_shift = (self.env.base_mass_low + self.env.base_mass_range * 
                          torch.rand(len(envs_idx), 1, device=self.env.device))
        other_mass_shift = (-self.env.other_mass_low + self.env.other_mass_range * 
                           torch.rand(len(envs_idx), self.env.robot.n_links - 1, 
                                     device=self.env.device))
        mass_shift = torch.cat((base_mass_shift, other_mass_shift), dim=1)
        self.env.robot.set_mass_shift(
            mass_shift=mass_shift,
            links_idx_local=np.arange(0, self.env.robot.n_links),
            envs_idx=envs_idx
        )
        
        # Center of mass randomization
        base_com_shift = (-self.env.domain_rand_cfg["random_base_com_shift"] / 2 + 
            self.env.domain_rand_cfg["random_base_com_shift"] * 
            torch.rand(len(envs_idx), 1, 3, device=self.env.device))
        other_com_shift = (-self.env.domain_rand_cfg["random_other_com_shift"] / 2 + 
            self.env.domain_rand_cfg["random_other_com_shift"] * 
            torch.rand(len(envs_idx), self.env.robot.n_links - 1, 3, device=self.env.device))
        com_shift = torch.cat((base_com_shift, other_com_shift), dim=1)
        self.env.robot.set_COM_shift(
            com_shift=com_shift,
            links_idx_local=np.arange(0, self.env.robot.n_links),
            envs_idx=envs_idx
        )

        # PD gain randomization
        kp_shift = ((self.env.kp_low + self.env.kp_range *
                    torch.rand(len(envs_idx), self.env.num_actions, device=self.env.device)) * 
                   torch.from_numpy(self.env.kp[0]).to(self.env.device))
        self.env.robot.set_dofs_kp(
            kp_shift.cpu().numpy(), self.env.motors_dof_idx, envs_idx=envs_idx)

        kv_shift = ((self.env.kv_low + self.env.kv_range *
                    torch.rand(len(envs_idx), self.env.num_actions, device=self.env.device)) * 
                   torch.from_numpy(self.env.kv[0]).to(self.env.device))
        self.env.robot.set_dofs_kv(
            kv_shift.cpu().numpy(), self.env.motors_dof_idx, envs_idx=envs_idx)

        # Joint angle randomization
        dof_pos_shift = (self.env.joint_angle_low + self.env.joint_angle_range * 
            torch.rand(len(envs_idx), self.env.num_actions,
                       device=self.env.device, dtype=torch.float))
        self.env.default_dof_pos[envs_idx] = dof_pos_shift + self.env.basic_default_dof_pos

        # Damping randomization with curriculum
        if self.env.is_damping_descent:
            mean_episode_length = self.env.episode_lengths[envs_idx].mean() / \
                (self.env.env_cfg["episode_length_s"] / self.env.dt)

            if mean_episode_length > self.env.damping_threshold:
                self.env.damping_base -= self.env.damping_step
                if self.env.damping_base < self.env.damping_min:
                    self.env.damping_base = self.env.damping_min
            else:
                self.env.damping_base += self.env.damping_step
                if self.env.damping_base > self.env.damping_max:
                    self.env.damping_base = self.env.damping_max

        damping = ((self.env.dof_damping_low + self.env.dof_damping_range *
                   torch.rand(len(envs_idx), self.env.robot.n_dofs, device=self.env.device)) * 
                  self.env.damping_base)
        damping[:, :6] = 0
        self.env.robot.set_dofs_damping(
            damping=damping.cpu().numpy(),
            dofs_idx_local=np.arange(0, self.env.robot.n_dofs),
            envs_idx=envs_idx)

        # Armature randomization
        armature = (self.env.dof_armature_low + self.env.dof_armature_range *
                    torch.rand(len(envs_idx), self.env.robot.n_dofs, device=self.env.device))
        armature[:, :6] = 0
        self.env.robot.set_dofs_armature(
            armature=armature.cpu().numpy(),
            dofs_idx_local=np.arange(0, self.env.robot.n_dofs),
            envs_idx=envs_idx)