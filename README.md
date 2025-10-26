# wheel_quadruped_genesis

[中文文档](README_zh.md)

## Overview

This repository contains a wheel‑quadruped locomotion simulator and reinforcement learning framework based on the [wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) project. It extends the original work to support Unitree wheel‑quadruped robots (B2W and GO2W) using the [Genesis simulator](https://github.com/eric-he98/genesis) and a custom reinforcement‑learning toolkit. The goal of this project is to train and evaluate wheel‑legged quadruped locomotion policies in simulation and provide the assets and scripts required to deploy on real robots or export the trained policy to ONNX for inference.

## Features

- **Simulation Environment**: Custom [WheelLeggedEnv](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_env.py#L12-L784) environment (in [locomotion/wheel_legged_env.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_env.py)) built on the Genesis physics engine, defining robot dynamics, contact handling, observation and reward functions with RL algorithm interfaces
- **Robot Models**: URDF/Xacro descriptions for Unitree B2W and GO2W wheel-quadruped robots located in `assets/b2w_description` and `assets/go2w_description`
- **Reinforcement Learning**: PPO/on-policy training pipeline implemented with the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library, featuring adjustable hyperparameters in [locomotion/wheel_legged_train.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_train.py) and support for multiple parallel environments for fast training
- **Evaluation & Deployment**:
  - Policy evaluation scripts ([locomotion/wheel_legged_eval.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_eval.py))
  - Environment testing ([model_test.py](file:///home/ember/GitHub/wheel_quadruped_genesis/locomotion/model_test.py))
  - ONNX export capability ([onnx/pt2onnx.py](file:///home/ember/GitHub/wheel_quadruped_genesis/onnx/pt2onnx.py)) for deployment
- **Visualization & Debugging**:
  - TensorBoard logging in `logs/` for monitoring training progress
  - Joystick/keyboard control support via [utils/gamepad.py](file:///home/ember/GitHub/wheel_quadruped_genesis/utils/gamepad.py)
- **Debug Support**: Debug logs and tuning notes in `调试寄路/` (Chinese) documenting common issues and solutions when modifying URDF and environment (e.g., handling fixed-joint collapsing and observation dimension mismatches)
- **Gamepad Teleoperation**: Gamepad remote control support (`rsl_rl/utils/gamepad.py`)
- **Trained Models**: Pre-trained models and training logs available in `logs/`
- **ONNX Export**: Script to convert trained policies to ONNX format for deployment ([onnx/pt2onnx.py](file:///home/ember/GitHub/wheel_quadruped_genesis/onnx/pt2onnx.py))

## Directory Structure

```text
wheel_quadruped_genesis/
├── assets/                 # URDF, xacro, and mesh assets for B2W & GO2W
├── locomotion/            # Training and simulation scripts
├── onnx/                  # Script to convert JIT models to ONNX
├── rsl_rl/                # Local copy of rsl_rl RL framework
├── 调试寄路/              # Debug logs and troubleshooting notes (中文)
├── logs/                  # TensorBoard logs of training runs
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

4. Compile URDF assets if needed:
   *(Optional: for visualization/rendering)*

## Training

To train a wheel-quadruped agent:

```bash
python locomotion/wheel_legged_train.py
```

Training logs will be saved in `logs/wheel-quadruped-walking/`.

To monitor training progress:

```bash
tensorboard --logdir logs/
```

## Evaluation

To run trained policy for evaluation:

```bash
python locomotion/model_test.py
```

Ensure trained weights exist in the log directory or adjust config path.

## Gamepad Control

You can use a keyboard/gamepad to control the robot in simulation:

```bash
python rsl_rl/utils/gamepad_test.py
```

## Export to ONNX

To export the trained policy (JIT format) to ONNX:

```bash
python onnx/pt2onnx.py
```

This produces `policy.onnx` that can be used in real-time inference engines.

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
