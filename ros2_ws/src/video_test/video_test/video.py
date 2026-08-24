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
        self.set_frame = 1
        cap = cv2.VideoCapture(self.video_path)
        
        self.max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total number of frames: {max_frames}")
        cap.release()
        
        
        
        self.send_video_path = self.create_publisher(String, "/test_video/video_path", 10)
        self.send_frame = self.create_publisher(INT32, "/test_video/frame", 10)
            
        self.get_frame = self.create_subscription(
            Image,
            '/test_video/image_raw',
            self.image_callback,
            10
        )
        
        self.give_model = self.create_publisher(
            Image,
            '/test_video/input_image',
            1
            
        )
        self.yolo_sub = self.create_subscription(
            Float32MultiArray,
            '/yolov11/detections',
            self.yolo_callback,
            1
        )

        self.laneatt_sub = self.create_subscription(
            Float32MultiArray,
            '/laneatt/detections',
            self.laneatt_callback,
            1
        )
        self.start_timer = self.create_timer(1.0, self.start_video)
        self.started = False
        
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
        msg,
        desired_encoding='bgr8')
        
        if self.frame_number >= self.max_frames:
            self.get_logger().info('Finished processing video.')
            return
            
        
        
        
        
        
        

        # `frame` is now a cv2 / NumPy BGR image again.
        # cv2.imshow('Received frame', frame)
        # cv2.waitKey(1)
        
        
        
        # self.get_logger().info('Received image message')
        
    def timer_callback(self):
        # This function will be called every time the timer expires
        # self.get_logger().info('Timer callback triggered')
    
      