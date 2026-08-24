import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class CenterNode(Node):
    def __init__(self):
        super().__init__('center_node')
        self.get_split_left = self.create_subscriber(Float32MultiArray, "/laneatt/left_lane", self.lane_callback,1)
        self.get_split_right = self.create_subscriber(Float32MultiArray, "/laneatt/right_lane", 1)
        
    def lane_callback(self, msg):
        self.get_logger().info(f'Split_left: Type {type(self.get_split_left)}')
        self.get_logger().info(f"Left: {self.get_split_left}")
        self.get_logger().info(f'Split_right: Type {type(self.get_split_right)}')
        self.get_logger().info(f"Right: {self.get_split_right}")
    
        
        
def main(args=None):
    rclpy.init(args=args)
    node = CenterNode()
    rclpy.spin(node)
    # node.trt_engine.close()
    # node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
