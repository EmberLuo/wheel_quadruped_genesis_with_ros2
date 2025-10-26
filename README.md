# wheel_quadruped_genesis

[中文文档](README_zh.md)

## Overview

This repository contains a wheel‑quadruped locomotion simulator and reinforcement learning framework based on the [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) project. It extends the original work to support Unitree wheel‑quadruped robots (B2W and GO2W) using the [Genesis simulator](https://github.com/eric-he98/genesis) and a custom reinforcement‑learning toolkit. The goal of this project is to train and evaluate wheel‑legged quadruped locomotion policies in simulation and provide the assets and scripts required to deploy on real robots or export the trained policy to ONNX for inference.

## Features

- **Simulation Environment**: Custom [WheelLeggedEnv](locomotion/wheel_legged_env/wheel_legged_env.py) environment built on the Genesis physics engine, defining robot dynamics, contact handling, observation and reward functions with RL algorithm interfaces
- **Robot Models**: URDF/Xacro descriptions for Unitree B2W and GO2W wheel-quadruped robots located in `assets/b2w_description` and `assets/go2w_description`
- **Reinforcement Learning**: PPO/on-policy training pipeline implemented with the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library, featuring adjustable hyperparameters in [locomotion/wheel_legged_train_ros2.py](locomotion/wheel_legged_train_ros2.py) and support for multiple parallel environments for fast training
- **Evaluation & Deployment**:
  - Policy evaluation scripts ([locomotion/wheel_legged_eval_ros2.py](locomotion/wheel_legged_eval_ros2.py))
  - Environment testing ([locomotion/model_test.py](locomotion/model_test.py))
  - ONNX export capability ([onnx/pt2onnx.py](onnx/pt2onnx.py)) for deployment
- **Visualization & Debugging**:
  - TensorBoard logging in `logs/` for monitoring training progress
  - Joystick/keyboard control support via [utils/gamepad.py](utils/gamepad.py)
- **Debug Support**: Debug logs and tuning notes in `调试寄路/` (Chinese) documenting common issues and solutions when modifying URDF and environment (e.g., handling fixed-joint collapsing and observation dimension mismatches)
- **Gamepad Teleoperation**: Gamepad remote control support (`utils/gamepad.py`)
- **Trained Models**: Pre-trained models and training logs available in `logs/`
- **ONNX Export**: Script to convert trained policies to ONNX format for deployment ([onnx/pt2onnx.py](onnx/pt2onnx.py))
- **ROS2 Integration**: ROS2 support for robot control and communication (see [ROS2 Integration](#ros2-integration) section)

## Directory Structure

```text
wheel_quadruped_genesis/
├── assets/                 # URDF, xacro, and mesh assets for B2W & GO2W
├── locomotion/            # Training and simulation scripts
│   ├── wheel_legged_env/  # Environment implementation
│   ├── wheel_legged_train_ros2.py  # Training script with ROS2 support
│   └── wheel_legged_eval_ros2.py   # Evaluation script with ROS2 support
├── onnx/                  # Script to convert JIT models to ONNX
├── rsl_rl/                # Local copy of rsl_rl RL framework
├── ros2_ws/               # ROS2 workspace
│   └── src/
│       └── wheel_quadruped_ros/  # ROS2 package
│           ├── msg/       # ROS2 message definitions
│           ├── srv/       # ROS2 service definitions
│           ├── scripts/   # ROS2 node scripts
│           └── launch/    # ROS2 launch files
├── logs/                  # TensorBoard logs of training runs
└── utils/                 # Utility scripts (gamepad control, etc.)
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/EmberLuo/wheel_quadruped_genesis.git
   ```

   ```bash
   cd wheel_quadruped_genesis
   ```

2. Create conda environment:
   *(Adjust Python version as needed; tested on Python 3.1x with Genesis)*

   ```bash
   conda create -n genesis python=3.1x
   ```

   ```bash
   conda activate genesis
   ```

3. Install dependencies:
   *(Assumes genesis installed separately)*

   ```bash
   pip install -e rsl_rl
   ```

   ```bash
   pip install torch torchvision
   ```

   ```bash
   pip install gym numpy matplotlib onnx
   ```

4. (Optional) Install ROS2 for robot control:
   *(Required only if using ROS2 integration)*

   ```bash
   # Ubuntu 22.04 (Humble Hawksbill)
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
   sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
   sudo apt update
   sudo apt install ros-humble-desktop python3-argcomplete
   sudo apt install python3-colcon-common-extensions
   pip install rclpy
   ```

   ```bash
   # Build ROS2 workspace
   cd ros2_ws
   colcon build --packages-select wheel_quadruped_ros
   source install/setup.bash
   ```

5. Compile URDF assets if needed:
   *(Optional: for visualization/rendering)*

## Training

To train a wheel-quadruped agent:

```bash
python locomotion/wheel_legged_train_ros2.py
```

For training without ROS2 integration:

```bash
python locomotion/wheel_legged_train_ros2.py --no_ros2
```

Training logs will be saved in `logs/wheel-quadruped-walking/`.

To monitor training progress:

```bash
tensorboard --logdir logs/
```

## Evaluation

To run trained policy for evaluation:

```bash
python locomotion/wheel_legged_eval_ros2.py
```

For evaluation without ROS2 integration:

```bash
python locomotion/wheel_legged_eval_ros2.py --no_ros2
```

Alternatively, use the model test script:

```bash
python locomotion/model_test.py
```

Ensure trained weights exist in the log directory or adjust config path.

## Gamepad Control

You can use a keyboard/gamepad to control the robot in simulation:

```bash
python locomotion/wheel_legged_eval_ros2.py
```

The script will automatically detect and use the gamepad if available.

## Export to ONNX

To export the trained policy (JIT format) to ONNX:

```bash
python onnx/pt2onnx.py
```

This produces `policy.onnx` that can be used in real-time inference engines.

## ROS2 Integration

This project includes ROS2 support for robot control and communication. The ROS2 integration provides:

- **Robot State Publishing**: Real-time robot state information
- **Command Reception**: Remote control through ROS2 topics
- **Training Control**: Start/stop training through ROS2 services

### Using ROS2 for Training

To train with ROS2 integration:

```bash
python locomotion/wheel_legged_train_ros2.py --use_ros2
```

### Using ROS2 for Evaluation

To evaluate with ROS2 integration:

```bash
python locomotion/wheel_legged_eval_ros2.py --use_ros2
```

### Launching ROS2 Nodes

You can also launch the ROS2 nodes directly:

```bash
cd ros2_ws
source install/setup.bash
ros2 launch wheel_quadruped_ros wheel_quadruped.launch.py
```

### ROS2 Topics and Services

- **Topics**:
  - `/robot_state`: Publishes robot state information
  - `/robot_command`: Receives robot control commands

- **Services**:
  - `/train_control`: Controls training process
  - `/eval_control`: Controls evaluation process

### Example Control Commands

To send commands to the robot:

```bash
# Example using ros2 topic pub
ros2 topic pub /robot_command wheel_quadruped_ros/msg/RobotCommand "{header: {stamp: {sec: 0}, frame_id: ''}, velocity_command: {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}, joint_positions: [0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0], control_mode: 'policy'}"
```

## Known Issues

- URDF/Xacro parsing in Genesis may cause joint folding or DOF mismatch; see `调试寄路/记录1.md`
- Some `.pyc`, `build/`, and `CMakeFiles/` folders should be cleaned in production

## Credits

This repository is adapted from:

- [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis)
- [Genesis simulator](https://github.com/eric-he98/genesis)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

URDF models based on:

- Unitree B2/GO2 specifications (B2W & GO2W)

## License

MIT License (if not specified otherwise by upstream projects)
