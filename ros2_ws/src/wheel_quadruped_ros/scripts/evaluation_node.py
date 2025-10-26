#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import torch
import numpy as np
import pickle
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from wheel_quadruped_ros.msg import RobotState, RobotCommand
from wheel_quadruped_ros.srv import TrainControl
from locomotion.wheel_legged_env.wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', '..', 'locomotion'))


class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')

        # 创建回调组
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()

        # 创建发布者和订阅者
        self.state_publisher = self.create_publisher(
            RobotState, 'robot_state', 10)
        self.command_subscription = self.create_subscription(
            RobotCommand, 'robot_command', self.command_callback, 10)

        # 创建服务
        self.eval_control_service = self.create_service(
            TrainControl, 'eval_control', self.eval_control_callback,
            callback_group=self.service_callback_group)

        # 创建定时器
        self.timer = self.create_timer(0.01, self.timer_callback,
                                       callback_group=self.timer_callback_group)

        # 初始化变量
        self.policy = None
        self.env = None
        self.is_running = False

        self.get_logger().info('Evaluation node initialized')

    def init_environment(self, exp_name, ckpt):
        """初始化评估环境"""
        try:
            # 加载环境配置
            log_dir = f"logs/{exp_name}"
            env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(
                open(f"{log_dir}/cfgs.pkl", "rb"))

            # 创建环境
            self.env = WheelLeggedEnv(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                curriculum_cfg=curriculum_cfg,
                domain_rand_cfg=domain_rand_cfg,
                terrain_cfg=terrain_cfg,
                show_viewer=True,
                train_mode=False
            )

            # 加载模型
            runner = OnPolicyRunner(
                self.env, train_cfg, log_dir, device="cuda:0")
            resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
            runner.load(resume_path)
            self.policy = runner.get_inference_policy(device="cuda:0")

            self.get_logger().info('Environment and policy initialized successfully')
            return True

        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize environment: {str(e)}')
            return False

    def command_callback(self, msg):
        """处理接收到的机器人命令"""
        if self.env is not None:
            # 更新命令
            commands = np.array([
                msg.cmd_vel.linear.x,
                msg.cmd_vel.linear.y,
                msg.cmd_vel.angular.z
            ])
            self.env.set_commands(0, commands)

    def eval_control_callback(self, request, response):
        """处理评估控制服务请求"""
        if request.start:
            if self.init_environment(request.exp_name, 1300):
                self.is_running = True
                response.success = True
                response.message = "Evaluation started successfully"
            else:
                response.success = False
                response.message = "Failed to start evaluation"
        else:
            self.is_running = False
            response.success = True
            response.message = "Evaluation stopped"

        return response

    def timer_callback(self):
        """定时器回调函数，执行评估循环"""
        if not self.is_running or self.policy is None or self.env is None:
            return

        try:
            with torch.no_grad():
                # 执行策略
                actions = self.policy(self.env.obs)
                obs, _, rews, dones, infos = self.env.step(actions)

                # 发布机器人状态
                state_msg = RobotState()
                state_msg.header = Header()
                state_msg.header.stamp = self.get_clock().now().to_msg()

                # 填充关节状态
                joint_state = JointState()
                joint_state.header = state_msg.header
                joint_state.position = infos[0]['joint_pos'].cpu(
                ).numpy().tolist()
                joint_state.velocity = infos[0]['joint_vel'].cpu(
                ).numpy().tolist()
                state_msg.joint_states = joint_state

                # 填充IMU数据
                imu_msg = Imu()
                imu_msg.header = state_msg.header
                imu_msg.orientation = infos[0]['base_quat'].cpu(
                ).numpy().tolist()
                imu_msg.angular_velocity = infos[0]['base_ang_vel'].cpu(
                ).numpy().tolist()
                state_msg.imu = imu_msg

                # 填充速度信息
                twist = Twist()
                twist.linear.x = infos[0]['base_lin_vel'][0].cpu().numpy()
                twist.linear.y = infos[0]['base_lin_vel'][1].cpu().numpy()
                twist.angular.z = infos[0]['base_ang_vel'][2].cpu().numpy()
                state_msg.base_velocity = twist

                self.state_publisher.publish(state_msg)

                if dones:
                    self.env.reset()

        except Exception as e:
            self.get_logger().error(f'Error in evaluation loop: {str(e)}')
            self.is_running = False


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
