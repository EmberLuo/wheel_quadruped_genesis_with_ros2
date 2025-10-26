# ROS系统架构详解

## 1. ROS消息定义

### RobotCommand.msg

```python
# Command to send to the robot
std_msgs/Header header

# Velocity commands
geometry_msgs/Twist velocity_command

# Joint position commands
float32[] joint_positions

# Control mode
string control_mode
```

**功能介绍**：

- 这是机器人控制命令的消息定义
- 包含速度命令、关节位置命令和控制模式
- 用于向机器人发送控制指令

**输入输出**：

- 输入：由其他节点发布，包含机器人的控制指令
- 输出：被订阅者节点接收，用于控制机器人行为

### RobotState.msg

```python
# Robot state information
std_msgs/Header header

# Base pose
geometry_msgs/Pose base_pose
geometry_msgs/Twist base_velocity

# Joint states
sensor_msgs/JointState joint_states

# Contact information
float32[] contact_forces

# Command information
geometry_msgs/Twist command
```

**功能介绍**：

- 机器人状态信息的消息定义
- 包含机器人基座姿态、速度、关节状态、接触力和命令信息
- 用于发布机器人的当前状态

**输入输出**：

- 输入：由机器人节点根据实际状态填充
- 输出：发布给其他节点，用于监控和反馈

### TrainControl.srv

```python
# Request
string command  # start, stop, pause, resume
string parameter  # additional parameters

---
# Response
bool success
string message
```

**功能介绍**：

- 训练控制服务的定义
- 用于控制训练过程的启动、停止、暂停和恢复

**输入输出**：

- 输入：命令字符串和附加参数
- 输出：操作成功标志和消息

## 2. ROS节点实现

### wheel_quadruped_node.py

**功能介绍**：

- 核心机器人控制节点
- 负责处理机器人命令、发布机器人状态和提供训练控制服务
- 与底层环境进行交互

**关键代码分析**：

```python
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
```

**输入输出**：

- 输入：
  - 通过`robot_command`话题接收RobotCommand消息
  - 通过`train_control`服务接收TrainControl请求
- 输出：
  - 通过`robot_state`话题发布RobotState消息
  - 通过`train_control`服务返回TrainControl响应

### training_node.py

**功能介绍**：

- 训练过程管理节点
- 负责启动机器人训练过程
- 与训练逻辑集成

**关键代码分析**：

```python
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
```

**输入输出**：

- 输入：通过参数或服务调用接收训练配置
- 输出：启动训练过程并记录日志

### evaluation_node.py

**功能介绍**：

- 评估过程管理节点
- 负责加载训练好的模型并进行评估
- 提供评估控制和状态发布功能

**关键代码分析**：

```python
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
```

**输入输出**：

- 输入：
  - 通过`robot_command`话题接收RobotCommand消息
  - 通过`eval_control`服务接收TrainControl请求
- 输出：
  - 通过`robot_state`话题发布RobotState消息
  - 通过`eval_control`服务返回TrainControl响应

## 3. 训练与评估逻辑集成

### wheel_legged_train_ros2.py

**功能介绍**：

- 支持ROS2的训练脚本
- 提供训练配置和环境初始化
- 可以在ROS2环境中运行训练过程

**关键代码分析**：

```python
def train_with_ros2(exp_name="wheel-quadruped-walking", num_envs=1024, max_iterations=7000):
    """支持ROS2的训练函数"""
    gs.init(logging_level="warning", backend=gs.gpu)

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name, max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg,
        show_viewer=False,
        num_view=10,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg,
            domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=max_iterations,
                 init_at_random_ep_len=True)
```

**输入输出**：

- 输入：训练参数（实验名称、环境数量、最大迭代次数）
- 输出：训练好的模型和配置文件

### wheel_legged_eval_ros2.py

**功能介绍**：

- 支持ROS2的评估脚本
- 加载训练好的模型并进行评估
- 可以通过ROS2或游戏手柄进行控制

**关键代码分析**：

```python
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
```

**输入输出**：

- 输入：
  - 模型检查点参数
  - 通过ROS2话题接收控制命令
- 输出：
  - 通过ROS2话题发布机器人状态
  - 可视化界面

## 4. 启动配置

### wheel_quadruped.launch.py

**功能介绍**：

- ROS2启动文件
- 配置和启动wheel_quadruped_node节点
- 提供参数配置接口

**关键代码分析**：

```python
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
```

**输入输出**：

- 输入：启动参数（实验名称、环境数量、最大迭代次数）
- 输出：配置好的wheel_quadruped_node节点

## 5. 系统整体工作流程

1. **训练阶段**：
   - 使用`wheel_legged_train_ros2.py`启动训练过程
   - 训练过程中生成模型和配置文件
   - 可通过ROS2服务控制训练过程

2. **评估阶段**：
   - 使用`wheel_legged_eval_ros2.py`加载训练好的模型
   - 可以通过ROS2或游戏手柄进行控制
   - 评估过程中发布机器人状态

3. **实时控制**：
   - `wheel_quadruped_node.py`作为核心控制节点
   - 接收控制命令并执行
   - 发布机器人状态信息

## 6. ROS通信机制

1. **话题通信**：
   - `robot_command`：发布机器人控制命令
   - `robot_state`：发布机器人状态信息

2. **服务通信**：
   - `train_control`：控制训练过程
   - `eval_control`：控制评估过程

3. **参数传递**：
   - 通过启动文件和ROS2参数服务器传递配置参数

## 总结

这个代码仓库中的ROS系统主要用于轮式四足机器人的训练、评估和实时控制。通过ROS2的消息、服务和参数机制，实现了训练过程管理、模型评估和实时控制的集成。系统采用模块化设计，各节点通过标准ROS接口进行通信，便于扩展和维护。

核心功能包括：

1. 训练过程的启动和控制
2. 训练好的模型加载和评估
3. 机器人状态的实时监控
4. 通过ROS2接口进行机器人控制

这种设计使得训练和评估过程可以通过ROS2进行远程控制和监控，提高了系统的灵活性和可扩展性。
