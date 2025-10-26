import torch

class TerminationManager:
    def __init__(self, env):
        self.env = env

    def check_termination(self):
        """Check termination conditions."""
        # Timeout termination
        self.env.reset_buf = self.env.episode_length_buf > self.env.max_episode_length

        # Pitch angle termination
        self.env.reset_buf |= torch.abs(
            self.env.base_euler[:, 1]) > self.env.env_cfg["termination_if_pitch_greater_than"]

        # Roll angle termination
        self.env.reset_buf |= torch.abs(
            self.env.base_euler[:, 0]) > self.env.env_cfg["termination_if_roll_greater_than"]

        # Base contact termination
        if self.env.env_cfg["termination_if_base_contact_plane_than"]:
            for idx in self.env.reset_links:
                self.env.reset_buf |= torch.abs(
                    self.env.connect_force[:, idx, :]).sum(dim=1) > 0