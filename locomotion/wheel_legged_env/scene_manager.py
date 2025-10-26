import torch
import numpy as np
import cv2
import genesis as gs
from genesis.utils.geom import inv_quat

class SceneManager:
    def __init__(self, env):
        self.env = env

    def create_scene(self, show_viewer, num_view):
        """Create the simulation scene."""
        self.env.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.env.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.env.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(num_view))),
            rigid_options=gs.options.RigidOptions(
                dt=self.env.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane
        self.env.scene.add_entity(gs.morphs.URDF(
            file="assets/terrain/plane/plane.urdf", fixed=True))

    def add_terrain(self):
        """Add terrain to the simulation."""
        # Initialize base position and orientation
        self.env.base_init_pos = torch.tensor(
            self.env.env_cfg["base_init_pos"]["urdf"], device=self.env.device)
        self.env.base_init_quat = torch.tensor(
            self.env.env_cfg["base_init_quat"], device=self.env.device)
        self.env.inv_base_init_quat = inv_quat(self.env.base_init_quat)
        self.env.terrain_height = torch.zeros((1, 1), device=self.env.device)
        # Terrain parameters
        self.env.horizontal_scale = self.env.terrain_cfg["horizontal_scale"]
        self.env.vertical_scale = self.env.terrain_cfg["vertical_scale"]

        if self.env.terrain_cfg["terrain"]:
            print("\033[1;35m open terrain\033[0m")
            if self.env.mode:
                # Training terrain
                height_field_path = "assets/terrain/png/" + \
                    self.env.terrain_cfg["train"] + ".png"
                self.env.height_field = cv2.imread(
                    height_field_path, cv2.IMREAD_GRAYSCALE)
                self.env.terrain_height = torch.tensor(
                    self.env.height_field, device=self.env.device) * self.env.vertical_scale

                self.env.terrain = self.env.scene.add_entity(
                    morph=gs.morphs.Terrain(
                        height_field=self.env.height_field,
                        horizontal_scale=self.env.horizontal_scale,
                        vertical_scale=self.env.vertical_scale,
                    ))

                # Respawn points
                self.env.base_terrain_pos = torch.zeros(
                    (self.env.num_respawn_points, 3), device=self.env.device)
                for i in range(self.env.num_respawn_points):
                    self.env.base_terrain_pos[i] = self.env.base_init_pos + \
                        torch.tensor(
                            self.env.respawn_points[i], device=self.env.device)
                print("\033[1;34m respawn_points: \033[0m",
                      self.env.base_terrain_pos)
            else:
                # Evaluation terrain
                height_field_path = "assets/terrain/png/" + \
                    self.env.terrain_cfg["eval"] + ".png"
                height_field = cv2.imread(
                    height_field_path, cv2.IMREAD_GRAYSCALE)
                self.env.terrain = self.env.scene.add_entity(
                    morph=gs.morphs.Terrain(
                        pos=(1.0, 1.0, 0.0),
                        height_field=height_field,
                        horizontal_scale=self.env.horizontal_scale,
                        vertical_scale=self.env.vertical_scale,
                    ))
                print("\033[1;34m respawn_points: \033[0m", self.env.base_init_pos)

    def add_robot(self, robot_morphs):
        """Add robot to the simulation."""
        base_init_pos = self.env.base_init_pos.cpu().numpy()
        if self.env.terrain_cfg["terrain"] and self.env.mode:
            base_init_pos = self.env.base_terrain_pos[0].cpu().numpy()

        # Add robot based on morphology type
        if robot_morphs == "urdf":
            self.env.robot = self.env.scene.add_entity(
                gs.morphs.URDF(
                    file=self.env.env_cfg["urdf"],
                    pos=base_init_pos,
                    quat=self.env.base_init_quat.cpu().numpy(),
                    convexify=self.env.env_cfg["convexify"],
                    decimate_aggressiveness=self.env.env_cfg["decimate_aggressiveness"],
                ))
        elif robot_morphs == "mjcf":
            self.env.robot = self.env.scene.add_entity(
                gs.morphs.MJCF(
                    file=self.env.env_cfg["mjcf"],
                    pos=base_init_pos,
                    quat=self.env.base_init_quat.cpu().numpy(),
                    convexify=self.env.env_cfg["convexify"],
                    decimate_aggressiveness=self.env.env_cfg["decimate_aggressiveness"],
                ),
                vis_mode='collision'
            )
        else:
            raise Exception(
                "Unknown robot morphology. Should be 'urdf' or 'mjcf'")

    def build_scene(self):
        """Build the simulation scene with all entities."""
        self.env.scene.build(n_envs=self.env.num_envs)

        # Create joint indices mapping
        self.env.motors_dof_idx = [self.env.robot.get_joint(name).dof_start
                                   for name in self.env.env_cfg["joint_names"]]

        joint_dof_idx = []
        wheel_dof_idx = []
        self.env.joint_dof_idx = []
        self.env.wheel_dof_idx = []

        for i in range(len(self.env.env_cfg["joint_names"])):
            if self.env.env_cfg["joint_type"][self.env.env_cfg["joint_names"][i]] == "joint":
                joint_dof_idx.append(i)
                self.env.joint_dof_idx.append(self.env.motors_dof_idx[i])
            elif self.env.env_cfg["joint_type"][self.env.env_cfg["joint_names"][i]] == "wheel":
                wheel_dof_idx.append(i)
                self.env.wheel_dof_idx.append(self.env.motors_dof_idx[i])

        self.env.joint_dof_idx_np = np.array(joint_dof_idx)
        self.env.wheel_dof_idx_np = np.array(wheel_dof_idx)

        # Initialize PD control parameters
        self._init_pd_control()

    def _init_pd_control(self):
        """Initialize PD control parameters."""
        # Proportional gains
        self.env.kp = np.full((self.env.num_envs, self.env.num_actions),
                              self.env.env_cfg["joint_kp"])
        self.env.kp[:, self.env.wheel_dof_idx_np] = 0.0

        # Derivative gains
        self.env.kv = np.full((self.env.num_envs, self.env.num_actions),
                              self.env.env_cfg["joint_kv"])
        self.env.kv[:, self.env.wheel_dof_idx_np] = self.env.env_cfg["wheel_kv"]

        self.env.robot.set_dofs_kp(self.env.kp, self.env.motors_dof_idx)
        self.env.robot.set_dofs_kv(self.env.kv, self.env.motors_dof_idx)

        # Damping parameters
        damping = np.full((self.env.num_envs, self.env.robot.n_dofs),
                          self.env.env_cfg["damping"])
        damping[:, :6] = 0  # First 6 DOFs are base DOFs

        self.env.is_damping_descent = self.env.curriculum_cfg["damping_descent"]
        damping_params = self.env.curriculum_cfg["dof_damping_descent"]
        self.env.damping_max = damping_params[0]
        self.env.damping_min = damping_params[1]
        self.env.damping_step = damping_params[2] * \
            (self.env.damping_max - self.env.damping_min)
        self.env.damping_threshold = damping_params[3]

        if self.env.is_damping_descent:
            self.env.damping_base = self.env.damping_max
        else:
            self.env.damping_base = self.env.env_cfg["damping"]

        self.env.robot.set_dofs_damping(damping, np.arange(0, self.env.robot.n_dofs))

        # Armature parameters
        armature = np.full((self.env.num_envs, self.env.robot.n_dofs),
                           self.env.env_cfg["armature"])
        armature[:, :6] = 0
        self.env.robot.set_dofs_armature(armature, np.arange(0, self.env.robot.n_dofs))