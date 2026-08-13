from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description = Command([
        'xacro ',
        PathJoinSubstitution([
            FindPackageShare('robot_model_pkg'),
            'urdf',
            'robot.xacro',
        ]),
    ])

    gazebo_launch = PathJoinSubstitution([
        FindPackageShare('gazebo_ros'),
        'launch',
        'gazebo.launch.py',
    ])

    world = PathJoinSubstitution([
        FindPackageShare('autonomous_bicycle_gazebo'),
        'worlds',
        'curved.world',
    ])

    gazebo_model_path = PathJoinSubstitution([
        FindPackageShare('autonomous_bicycle_gazebo'),
        'models',
    ])

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        SetEnvironmentVariable('QT_X11_NO_MITSHM', '1'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                'world': world,
                'verbose': 'true',
            }.items(),
        ),
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
                '-entity', 'drive_car',
                '-topic', 'robot_description',
                '-x', '-35',
                '-y', '0',
                '-z', '0.30',
            ],
            output='screen',
        ),
    ])
