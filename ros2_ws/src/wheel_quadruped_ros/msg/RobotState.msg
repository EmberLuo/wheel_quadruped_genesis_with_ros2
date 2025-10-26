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