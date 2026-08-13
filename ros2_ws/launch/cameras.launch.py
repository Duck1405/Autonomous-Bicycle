from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cameras = [
        {'name': 'front', 'device': '/dev/video4', 'width': 640, 'height': 480},
        {'name': 'right', 'device': '/dev/video2', 'width': 640,  'height': 480},
        {'name': 'left',  'device': '/dev/video6', 'width': 640,  'height': 480},
        {'name': 'back',  'device': '/dev/video8', 'width': 640,  'height': 480},
    ]

    nodes = []
    for cam in cameras:
        nodes.append(Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name=f'{cam["name"]}_camera',
            namespace=f'camera/{cam["name"]}',
            parameters=[{
                'video_device': cam['device'],
                'pixel_format': 'mjpeg2rgb',
                'image_width': cam['width'],
                'image_height': cam['height'],
                'framerate': 30.0,
                'camera_name': f'{cam["name"]}_camera',
            }],
        ))

    return LaunchDescription(nodes)
