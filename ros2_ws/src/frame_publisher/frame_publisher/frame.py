import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from std_msgs.msg import String


class Framepub(Node):
    def __init__(self):
        super().__init__("frameNodepub")
        
        self.video_path = "/home/mlc/aman/Autonomous-Bicycle/Videos2/IMG_6893_30fps.mp4"
        self.image_pub = self.create_publisher(Image, '/test_video/image_raw', 10)

        self.next_frame_sub = self.create_subscription(
            Empty,
            '/test_video/request_next_frame',
            self.send_next_frame,
            10
        )
        self.set_video_path = self.create_subscription()
        
        
        
        
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open: {self.video_path}')

        
        self.cap = cv2.VideoCapture(self.video_path)
        self.bridge = CvBridge()
        
    def send_next_frame(self, _request):
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, _request)
        success, frame = self.cap.read()

        if not success:
            self.get_logger().info('Video finished.')
            self.cap.release()
            return


        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(image_msg)
        