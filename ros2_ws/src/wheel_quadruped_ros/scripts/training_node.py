#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'locomotion'))

from locomotion.wheel_legged_train_ros2 import main as train_main

class TrainingNode(Node):
    def __init__(self):
        super().__init__('training_node')
        self.get_logger().info('Training node initialized')
        
    def start_training(self, exp_name="wheel-quadruped-walking", num_envs=1024, max_iterations=7000):
        """启动训练过程"""
        # 这里可以调用原始的训练函数
        # 需要修改wheel_legged_train.py以支持ROS2参数传递
        self.get_logger().info(f'Starting training: {exp_name}')
        # train_main()  # 需要适配参数

def main(args=None):
    rclpy.init(args=args)
    
    node = TrainingNode()
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()