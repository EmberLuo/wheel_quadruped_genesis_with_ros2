import genesis as gs  # type: ignore
import numpy as np
import time
import cv2
import math
import torch
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

robot = scene.add_entity(
    gs.morphs.URDF(file="assets/go2w_description/urdf/go2w_description.urdf",
                   pos=(0.0, 0.0, 0.45),
                   quat=(1, 0, 0, 0)
                   ),
    # gs.morphs.MJCF(file="assets/mjcf/point_foot2/point_foot2.xml",
    # pos=(0.0, 0.0, 0.65)
    # ),
    # vis_mode='collision'
)

# height_field = cv2.imread("assets/terrain/png/stairs.png", cv2.IMREAD_GRAYSCALE)
# terrain_height = torch.tensor(height_field) * 0.1
# print(terrain_height.size())
# terrain = scene.add_entity(
#         morph=gs.morphs.Terrain(
#         # pos = (0.0,0.0,0.0),
#         height_field = height_field,
#         horizontal_scale=0.1,
#         vertical_scale=0.001,
#         ),
#     )

cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,
)
scene.build(n_envs=1)

for solver in scene.sim.solvers:
    if not isinstance(solver, RigidSolver):
        continue
    rigid_solver = solver

jnt_names = [
    "FL_hip_joint",    # Front Left hip joint
    "FR_hip_joint",    # Front Right hip joint
    "RL_hip_joint",    # Rear Left hip joint
    "RR_hip_joint",    # Rear Right hip joint
    "FL_thigh_joint",  # Front Left thigh joint
    "FR_thigh_joint",  # Front Right thigh joint
    "RL_thigh_joint",  # Rear Left thigh joint
    "RR_thigh_joint",  # Rear Right thigh joint
    "FL_calf_joint",   # Front Left calf joint
    "FR_calf_joint",   # Front Right calf joint
    "RL_calf_joint",   # Rear Left calf joint
    "RR_calf_joint",   # Rear Right calf joint
    "FL_foot_joint",   # Front Left foot joint (wheel)
    "FR_foot_joint",   # Front Right foot joint (wheel)
    "RL_foot_joint",   # Rear Left foot joint (wheel)
    "RR_foot_joint",   # Rear Right foot joint (wheel)
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]
robot.set_dofs_kp(
    kp=np.array([40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 40, 40, 40, 40]),
    dofs_idx_local=dofs_idx,
)
robot.set_dofs_kv(
    kv=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    dofs_idx_local=dofs_idx,
)
left_knee = robot.get_joint("FL_calf_joint")

print(robot.n_links)
link = robot.get_link("FL_calf")
print(link.idx)
link = robot.get_link("FR_calf")
print(link.idx)


# 渲染rgb、深度、分割掩码和法线图
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

# cam.start_recording()
scene.step()
while True:
    # link_pos = rigid_solver.get_links_pos([1,])
    # force = 100 * link_pos
    # rigid_solver.apply_links_external_force(
    #     force=force,
    #     links_idx=[1,],
    # )

    # 将控制命令修改为符合关节限制的值
    robot.control_dofs_position(
        np.array([
            0.0,   # FL_hip_joint: [-1.0472, 1.0472]
            -0.0,   # FR_hip_joint: [-1.0472, 1.0472]
            0.0,   # RL_hip_joint: [-1.0472, 1.0472]
            -0.0,   # RR_hip_joint: [-1.0472, 1.0472]
            0.8,   # FL_thigh_joint: [-1.5708, 3.4907]
            0.8,   # FR_thigh_joint: [-1.5708, 3.4907]
            0.8,   # RL_thigh_joint: [-0.5236, 4.5379]
            0.8,   # RR_thigh_joint: [-0.5236, 4.5379]
            -1.5,   # FL_calf_joint: [-2.7227, -0.83776]
            -1.5,   # FR_calf_joint: [-2.7227, -0.83776]
            -1.5,   # RL_calf_joint: [-2.7227, -0.83776]
            -1.5,   # RR_calf_joint: [-2.7227, -0.83776]
            0.0,   # FL_foot_joint: continuous (no limits)
            0.0,   # FR_foot_joint: continuous (no limits)
            0.0,   # RL_foot_joint: continuous (no limits)
            0.0    # RR_foot_joint: continuous (no limits)
        ]),
        dofs_idx,
    )
    scene.step()
    # print(robot.get_pos())
    # left_knee_pos = left_knee.get_pos()
    # print("left_knee_pos    ",left_knee_pos)
    # force = robot.get_links_net_contact_force()
    # dof_vel = robot.get_dofs_velocity()
    # print("dof_pos:",robot.get_dofs_position(dofs_idx))
    # time.sleep(0.1)
    cam.render()
# cam.stop_recording(save_to_filename='video.mp4', fps=60)
