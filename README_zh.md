# wheel_quadruped_genesis

[English Documentation](README.md)

## 概述

本仓库包含一个轮式四足机器人运动仿真器和强化学习框架，基于 [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) 项目开发。它扩展了原始工作以支持 Unitree 轮式四足机器人（B2W 和 GO2W），使用 [Genesis 仿真器](https://github.com/eric-he98/genesis)和自定义强化学习工具包。该项目的目标是在仿真环境中训练和评估轮式四足机器人的运动策略，并提供在真实机器人上部署或导出训练策略到 ONNX 格式进行推理所需的资源和脚本。

## 功能特性

- **仿真环境**：基于 Genesis 物理引擎构建的自定义 [WheelLeggedEnv](locomotion/wheel_legged_env/wheel_legged_env.py) 环境，定义了机器人动力学、接触处理、观测和奖励函数以及 RL 算法接口
- **机器人模型**：位于 `assets/b2w_description` 和 `assets/go2w_description` 目录下的 Unitree B2W 和 GO2W 轮式四足机器人 URDF/Xacro 描述文件
- **强化学习**：使用 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 库实现的 PPO/在线策略训练流水线，在 [locomotion/wheel_legged_train_ros2.py](locomotion/wheel_legged_train_ros2.py) 中具有可调节的超参数，并支持多个并行环境以实现快速训练
- **评估与部署**：
  - 策略评估脚本 ([locomotion/wheel_legged_eval_ros2.py](locomotion/wheel_legged_eval_ros2.py))
  - 环境测试 ([locomotion/model_test.py](locomotion/model_test.py))
  - ONNX 导出功能 ([onnx/pt2onnx.py](onnx/pt2onnx.py)) 用于部署
- **可视化与调试**：
  - 在 `logs/` 中记录 TensorBoard 日志以监控训练进度
  - 通过 [utils/gamepad.py](utils/gamepad.py) 支持游戏手柄/键盘控制
- **调试支持**：在 `调试寄路/` 目录中的调试日志和调优笔记（中文），记录了修改 URDF 和环境时的常见问题和解决方案（例如，处理固定关节折叠和观测维度不匹配）
- **游戏手柄遥控**：游戏手柄远程控制支持 (`utils/gamepad.py`)
- **训练模型**：预训练模型和训练日志可在 `logs/` 中获取
- **ONNX 导出**：将训练策略转换为 ONNX 格式的脚本 ([onnx/pt2onnx.py](onnx/pt2onnx.py))
- **ROS2 集成**：ROS2 支持用于机器人控制和通信（参见 [ROS2 集成](#ros2-集成) 部分）

## 目录结构

```text
wheel_quadruped_genesis/
├── assets/                 # B2W & GO2W 的 URDF, xacro 和网格资源
├── locomotion/             # 训练和仿真脚本
│   ├── wheel_legged_env/  # 环境实现
│   ├── wheel_legged_train_ros2.py  # 支持 ROS2 的训练脚本
│   └── wheel_legged_eval_ros2.py   # 支持 ROS2 的评估脚本
├── onnx/                   # 将 JIT 模型转换为 ONNX 的脚本
├── rsl_rl/                 # rsl_rl RL 框架的本地副本
├── ros2_ws/               # ROS2 工作空间
│   └── src/
│       └── wheel_quadruped_ros/  # ROS2 包
│           ├── msg/       # ROS2 消息定义
│           ├── srv/       # ROS2 服务定义
│           ├── scripts/   # ROS2 节点脚本
│           └── launch/    # ROS2 启动文件
├── logs/                   # 训练运行的 TensorBoard 日志
└── utils/                 # 实用脚本（游戏手柄控制等）
```

## 环境设置

1. 克隆仓库：

   ```bash
   git clone https://github.com/EmberLuo/wheel_quadruped_genesis.git
   ```

   ```bash
   cd wheel_quadruped_genesis
   ```

2. 创建 conda 环境：
   *(根据需要调整 Python 版本；在 Python 3.1x 和 Genesis 上测试)*

   ```bash
   conda create -n genesis python=3.1x
   ```

   ```bash
   conda activate genesis
   ```

3. 安装依赖项：
   *(假设 genesis 已单独安装)*

   ```bash
   pip install -e rsl_rl
   ```

   ```bash
   pip install torch torchvision
   ```

   ```bash
   pip install gym numpy matplotlib onnx
   ```

4. （可选）安装 ROS2 用于机器人控制：
   *(仅在使用 ROS2 集成时需要)*

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
   # 构建 ROS2 工作空间
   cd ros2_ws
   colcon build --packages-select wheel_quadruped_ros
   source install/setup.bash
   ```

5. 如需要，编译 URDF 资源：
   *(可选：用于可视化/渲染)*

## 训练

训练轮式四足代理：

```bash
python locomotion/wheel_legged_train_ros2.py
```

不使用 ROS2 集成进行训练：

```bash
python locomotion/wheel_legged_train_ros2.py --no_ros2
```

训练日志将保存在 `logs/wheel-quadruped-walking/` 中。

监控训练进度：

```bash
tensorboard --logdir logs/
```

## 评估

运行训练策略进行评估：

```bash
python locomotion/wheel_legged_eval_ros2.py
```

不使用 ROS2 集成进行评估：

```bash
python locomotion/wheel_legged_eval_ros2.py --no_ros2
```

或者使用模型测试脚本：

```bash
python locomotion/model_test.py
```

确保日志目录中存在训练权重或调整配置路径。

## 游戏手柄控制

您可以使用键盘/游戏手柄在仿真中控制机器人：

```bash
python locomotion/wheel_legged_eval_ros2.py
```

脚本将自动检测并使用游戏手柄（如果可用）。

## 导出到 ONNX

将训练策略（JIT 格式）导出到 ONNX：

```bash
python onnx/pt2onnx.py
```

这将生成可用于实时推理引擎的 `policy.onnx`。

## ROS2 集成

本项目包含 ROS2 支持，用于机器人控制和通信。ROS2 集成提供：

- **机器人状态发布**：实时机器人状态信息
- **命令接收**：通过 ROS2 话题进行远程控制
- **训练控制**：通过 ROS2 服务启动/停止训练

### 使用 ROS2 进行训练

使用 ROS2 集成进行训练：

```bash
python locomotion/wheel_legged_train_ros2.py --use_ros2
```

### 使用 ROS2 进行评估

使用 ROS2 集成进行评估：

```bash
python locomotion/wheel_legged_eval_ros2.py --use_ros2
```

### 启动 ROS2 节点

您也可以直接启动 ROS2 节点：

```bash
cd ros2_ws
source install/setup.bash
ros2 launch wheel_quadruped_ros wheel_quadruped.launch.py
```

### ROS2 话题和服务

- **话题**：
  - `/robot_state`：发布机器人状态信息
  - `/robot_command`：接收机器人控制命令

- **服务**：
  - `/train_control`：控制训练过程
  - `/eval_control`：控制评估过程

### 示例控制命令

向机器人发送命令：

```bash
# 使用 ros2 topic pub 示例
ros2 topic pub /robot_command wheel_quadruped_ros/msg/RobotCommand "{header: {stamp: {sec: 0}, frame_id: ''}, velocity_command: {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}, joint_positions: [0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0, 0.0, 0.7, -1.4, 0.0], control_mode: 'policy'}"
```

## 已知问题

- Genesis 中的 URDF/Xacro 解析可能导致关节折叠或自由度不匹配；参见 `调试寄路/记录1.md`
- 一些 `.pyc`、`build/` 和 `CMakeFiles/` 文件夹应在生产环境中清理

## 致谢

本仓库改编自：

- [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis)
- [Genesis](https://github.com/eric-he98/genesis)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

URDF 模型基于：

- Unitree B2/GO2 规格（B2W & GO2W）

## 许可证

MIT 许可证（如上游项目未另行规定）
