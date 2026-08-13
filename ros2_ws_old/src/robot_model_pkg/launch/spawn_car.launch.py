from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')
    entity = LaunchConfiguration('entity')

    robot_description = Command([
        'xacro ',
        PathJoinSubstitution([
            FindPackageShare('robot_model_pkg'),
            'urdf',
            'robot.xacro',
        ]),
    ])

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='-39.0'),
        DeclareLaunchArgument('y', default_value='-2.0'),
        DeclareLaunchArgument('z', default_value='0.30'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('entity', default_value='drive_car'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }],
            output='screen',
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', entity,
                '-topic', 'robot_description',
                '-x', x,
                '-y', y,
                '-z', z,
                '-Y', yaw,
            ],
            output='screen',
        ),
    ])
