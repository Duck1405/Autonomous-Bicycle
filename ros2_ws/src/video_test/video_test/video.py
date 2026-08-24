import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String, INT32

# This will be out centeral node




class VideoTestNode(Node):
    def __init__(self):
        super().__init__('video_test_node')
    
        self.video_path = "/home/mlc/aman/Autonomous-Bicycle/Videos2/IMG_6893_30fps.mp4"
        self.send_video_path = self.create_publisher(String, "/test_video/video_path", 10)
        self.send_frame = self.create_publisher(INT32, "/test_video/frame", 10)
        
        
        
    
    
        
        # self.yolo_sub = self.create_subscription(Image, '/yolov11/detections', self.image_callback, 10)
        # self.Lane_left_sub = self.create_subscription(Image, '/laneatt/left_lane', self.image_callback, 10)
        # self.Lane_right_sub = self.create_subscription(Image, '/laneatt/right_lane', self.image_callback, 10)
        
        
    def image_callback(self, msg):
        # self.get_logger().info('Received image message')
        
    def timer_callback(self):
        # This function will be called every time the timer expires
        # self.get_logger().info('Timer callback triggered')
    
      