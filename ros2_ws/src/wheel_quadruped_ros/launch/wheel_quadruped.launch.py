from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 声明参数
    exp_name_arg = DeclareLaunchArgument(
        'exp_name',
        default_value='wheel-quadruped-walking',
        description='Experiment name'
    )
    
    num_envs_arg = DeclareLaunchArgument(
        'num_envs',
        default_value='1024',
        description='Number of environments'
    )
    
    max_iterations_arg = DeclareLaunchArgument(
        'max_iterations',
        default_value='7000',
        description='Maximum training iterations'
    )
    
    # 创建节点
    wheel_quadruped_node = Node(
        package='wheel_quadruped_ros',
        executable='wheel_quadruped_node.py',
        name='wheel_quadruped_node',
        output='screen',
        parameters=[{
            'exp_name': LaunchConfiguration('exp_name'),
            'num_envs': LaunchConfiguration('num_envs'),
            'max_iterations': LaunchConfiguration('max_iterations'),
        }]
    )
    
    return LaunchDescription([
        exp_name_arg,
        num_envs_arg,
        max_iterations_arg,
        wheel_quadruped_node,
    ])