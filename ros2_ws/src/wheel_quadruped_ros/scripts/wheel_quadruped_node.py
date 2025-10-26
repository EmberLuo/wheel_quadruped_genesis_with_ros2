#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'locomotion'))

from locomotion.wheel_legged_env.wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from wheel_quadruped_ros.msg import RobotState, RobotCommand
from wheel_quadruped_ros.srv import TrainControl

class WheelQuadrupedNode(Node):
    def __init__(self):
        super().__init__('wheel_quadruped_node')
        
        # 创建回调组
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        
        # 创建发布者和订阅者
        self.state_publisher_ = self.create_publisher(RobotState, 'robot_state', 10)
        self.command_subscription_ = self.create_subscription(
            RobotCommand, 'robot_command', self.command_callback, 10)
        
        # 创建服务
        self.train_control_service_ = self.create_service(
            TrainControl, 'train_control', self.train_control_callback,
            callback_group=self.service_callback_group)
        
        # 创建定时器
        self.timer_ = self.create_timer(0.01, self.timer_callback, 
                                       callback_group=self.timer_callback_group)
        
        # 初始化环境
        self.init_environment()
        
        self.get_logger().info('Wheel Quadruped ROS node has been started')

    def init_environment(self):
        """初始化训练环境"""
        # 这里需要根据你的配置加载环境
        # 为了简化，我们使用默认配置
        env_cfg = {
            "num_actions": 16,
            # 其他配置参数...
        }
        
        obs_cfg = {
            "num_obs": 507,
            "num_slice_obs": 50,
            "history_length": 9,
            # 其他配置参数...
        }
        
        # 实际使用时需要加载完整的配置
        # self.env = WheelLeggedEnv(...)
        
        self.get_logger().info('Environment initialized')

    def command_callback(self, msg):
        """处理接收到的机器人命令"""
        self.get_logger().info(f'Received command: {msg.control_mode}')
        # 处理命令逻辑

    def train_control_callback(self, request, response):
        """处理训练控制服务请求"""
        command = request.command.lower()
        
        if command == 'start':
            response.success = True
            response.message = 'Training started'
        elif command == 'stop':
            response.success = True
            response.message = 'Training stopped'
        elif command == 'pause':
            response.success = True
            response.message = 'Training paused'
        elif command == 'resume':
            response.success = True
            response.message = 'Training resumed'
        else:
            response.success = False
            response.message = f'Unknown command: {command}'
            
        return response

    def timer_callback(self):
        """定时器回调函数，发布机器人状态"""
        # 创建机器人状态消息
        state_msg = RobotState()
        # 填充状态信息
        # 这里需要从实际环境中获取数据
        
        # 发布状态
        self.state_publisher_.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = WheelQuadrupedNode()
    
    # 使用多线程执行器
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