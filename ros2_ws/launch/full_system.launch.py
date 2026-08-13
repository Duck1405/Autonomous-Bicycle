import math
from launch import LaunchDescription
from launch_ros.actions import Node

def static_tf(name, x, y, z, yaw):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'tf_{name}',
        arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                   '--yaw', str(yaw), '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base_link', '--child-frame-id', f'{name}_camera_link']
    )

def generate_launch_description():
    cameras = [
        {'name': 'front', 'device': '/dev/video4', 'width': 640, 'height': 480},
        {'name': 'right', 'device': '/dev/video2', 'width': 640, 'height': 480},
        {'name': 'left',  'device': '/dev/video6', 'width': 640, 'height': 480},
        {'name': 'back',  'device': '/dev/video10', 'width': 640, 'height': 480},
    ]

    camera_nodes = []
    for cam in cameras:
        camera_nodes.append(Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name=f'{cam["name"]}_camera',
            namespace=f'camera/{cam["name"]}',
            parameters=[{
                'video_device': cam['device'],
                'pixel_format': 'mjpeg2rgb',
                'image_width': cam['width'],
                'image_height': cam['height'],
                'framerate': 15.0,
                'camera_name': f'{cam["name"]}_camera',
                'frame_id': f'{cam["name"]}_camera_link',
            }],
	   respawn=True,
	   respawn_delay=2.0,
        ))

    tf_nodes = [
        static_tf('front', 0.2, 0.0, 0.1, 0.0),
        static_tf('right', 0.0, -0.15, 0.1, -math.pi/2),
        static_tf('left',  0.0,  0.15, 0.1,  math.pi/2),
        static_tf('back', -0.2, 0.0, 0.1, math.pi),
        static_tf('stereo', 0.25, 0.0, 0.12, 0.0),  # ready for when CSI is working
    ]

    stereo_node = Node(
        package='imx219_83_jetson_driver',
        executable='camera_node',
        name='stereo_camera',
        parameters=[{
            'left_camera_id': 1,
            'right_camera_id': 0,
            'left_camera_name': 'stereo_left',
            'right_camera_name': 'stereo_right',
            'width': 1280,
            'height': 720,
            'framerate': '30/1',
            'flip_method': 2,
            'frame_id': 'stereo_camera_link',
        }],
    )

    return LaunchDescription(camera_nodes + tf_nodes + [stereo_node])
