import argparse
import os
import pickle

import torch
from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import gamepad
import copy

# 添加ROS2导入
try:
    import rclpy
    from rclpy.node import Node
    from ros2_ws.src.wheel_quadruped_ros.msg import RobotState, RobotCommand
    from ros2_ws.src.wheel_quadruped_ros.srv import TrainControl
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-quadruped-walking")
    parser.add_argument("--ckpt", type=int, default=1300)
    parser.add_argument("--use_ros2", action="store_true", help="Use ROS2 for communication")
    args = parser.parse_args()
    
    gs.init(backend=gs.gpu,logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym"
    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        robot_morphs="urdf",
        show_viewer=True,
        train_mode=False
    )
    print(reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
    torch.jit.script(model).save(log_dir+"/policy.pt")
    
    print("\n--- 模型加载测试 ---")
    try:
        loaded_policy = torch.jit.load(log_dir + "/policy.pt")
        loaded_policy.eval()
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    
    obs, _ = env.reset()
    
    if args.use_ros2 and ROS2_AVAILABLE:
        # 使用ROS2进行控制
        rclpy.init()
        node = Node('wheel_quadruped_eval')
        
        # 创建发布者和订阅者
        state_publisher = node.create_publisher(RobotState, 'robot_state', 10)
        command_subscription = node.create_subscription(
            RobotCommand, 'robot_command', lambda msg: handle_command(msg, env), 10)
        
        node.get_logger().info('Evaluation node with ROS2 started')
        
        with torch.no_grad():
            while rclpy.ok():
                actions = loaded_policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                
                # 发布机器人状态
                state_msg = RobotState()
                # 填充状态信息
                state_publisher.publish(state_msg)
                
                rclpy.spin_once(node, timeout_sec=0.01)
                
                if dones:
                    env.reset()
        
        rclpy.shutdown()
    else:
        # 使用游戏手柄控制
        pad = gamepad.control_gamepad(command_cfg,[1.2,0.3,6.0,0.03,0.03,1.0])
        with torch.no_grad():
            while True:
                actions = loaded_policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                comands,reset_flag = pad.get_commands()
                print(f"comands: {comands}")
                env.set_commands(0,comands)
                if reset_flag:
                    env.reset()


def handle_command(msg, env):
    """处理ROS2命令消息"""
    # 根据接收到的命令更新环境
    pass


if __name__ == "__main__":
    main()