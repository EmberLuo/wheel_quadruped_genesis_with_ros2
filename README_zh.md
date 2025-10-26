# wheel_quadruped_genesis

[English Documentation](README.md)

## 概述

本仓库包含一个轮式四足机器人运动仿真器和强化学习框架，基于 [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) 项目开发。它扩展了原始工作以支持 Unitree 轮式四足机器人（B2W 和 GO2W），使用 [Genesis 仿真器](https://github.com/eric-he98/genesis)和自定义强化学习工具包。该项目的目标是在仿真环境中训练和评估轮式四足机器人的运动策略，并提供在真实机器人上部署或导出训练策略到 ONNX 格式进行推理所需的资源和脚本。

## 功能特性

- **仿真环境**：基于 Genesis 物理引擎构建的自定义 [WheelLeggedEnv](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_env.py#L12-L784) 环境（在 [locomotion/wheel_legged_env.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_env.py) 中），定义了机器人动力学、接触处理、观测和奖励函数以及 RL 算法接口
- **机器人模型**：位于 `assets/b2w_description` 和 `assets/go2w_description` 目录下的 Unitree B2W 和 GO2W 轮式四足机器人 URDF/Xacro 描述文件
- **强化学习**：使用 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 库实现的 PPO/在线策略训练流水线，在 [locomotion/wheel_legged_train.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_train.py) 中具有可调节的超参数，并支持多个并行环境以实现快速训练
- **评估与部署**：
  - 策略评估脚本 ([locomotion/wheel_legged_eval.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_eval.py))
  - 环境测试 ([model_test.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/model_test.py))
  - ONNX 导出功能 ([onnx/pt2onnx.py](file:///home/ember/GitHub/wheel_quadruped_genesis/onnx/pt2onnx.py)) 用于部署
- **可视化与调试**：
  - 在 `logs/` 中记录 TensorBoard 日志以监控训练进度
  - 通过 [utils/gamepad.py](file:///home/ember/GitHub/wheel_quadruped_genesis/utils/gamepad.py) 支持游戏手柄/键盘控制
- **调试支持**：在 `调试寄路/` 目录中的调试日志和调优笔记（中文），记录了修改 URDF 和环境时的常见问题和解决方案（例如，处理固定关节折叠和观测维度不匹配）
- **游戏手柄遥控**：游戏手柄远程控制支持 (`rsl_rl/utils/gamepad.py`)
- **训练模型**：预训练模型和训练日志可在 `logs/` 中获取
- **ONNX 导出**：将训练策略转换为 ONNX 格式的脚本 ([onnx/pt2onnx.py](file:///home/ember/GitHub/wheel_quadruped_genesis/onnx/pt2onnx.py))

## 目录结构

```text
wheel_quadruped_genesis/
├── assets/                 # B2W & GO2W 的 URDF, xacro 和网格资源
├── locomotion/             # 训练和仿真脚本
├── onnx/                   # 将 JIT 模型转换为 ONNX 的脚本
├── rsl_rl/                 # rsl_rl RL 框架的本地副本
├── 调试寄路/               # 调试日志和故障排除笔记（中文）
├── logs/                   # 训练运行的 TensorBoard 日志
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

4. 如需要，编译 URDF 资源：
   *(可选：用于可视化/渲染)*

## 训练

训练轮式四足代理：

```bash
python locomotion/wheel_legged_train.py
```

训练日志将保存在 `logs/wheel-quadruped-walking/` 中。

监控训练进度：

```bash
tensorboard --logdir logs/
```

## 评估

运行训练策略进行评估：

```bash
python locomotion/model_test.py
```

确保日志目录中存在训练权重或调整配置路径。

## 游戏手柄控制

您可以使用键盘/游戏手柄在仿真中控制机器人：

```bash
python rsl_rl/utils/gamepad_test.py
```

## 导出到 ONNX

将训练策略（JIT 格式）导出到 ONNX：

```bash
python onnx/pt2onnx.py
```

这将生成可用于实时推理引擎的 `policy.onnx`。

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
