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
        self.get_split_left = self.create_subscription(Float32MultiArray, "/laneatt/left_lane", self.left_lane_callback,1)
        self.get_split_right = self.create_subscription(Float32MultiArray, "/laneatt/right_lane", self.right_lane_callback, 1)
        
        self.left_lane_pts = None
        self.right_lane_pts = None
    
    def left_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            # Reshape flat [x, y, x, y...] array back into (N, 2) coordinates
            self.left_lane_pts = data.reshape(-1, 2)
        else:
            self.left_lane_pts = None
            
            
    def right_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.right_lane_pts = data.reshape(-1, 2)
        else:
            self.right_lane_pts = None
        
        self.get_center()
    
    def get_center(self):
        if self.right_lane_pts and self.left_lane_pts:
            self.get_logger().info("Both Found")
        elif self.right_lane_pts == None:
            self.get_logger().info("Right Lane Not Found")
        else:
            self.get_logger().info("Left Lane Not Found")
            
            
            
    
        
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
