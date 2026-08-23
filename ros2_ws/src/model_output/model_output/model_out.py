import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time










def main(args=None):
    rclpy.init(args=args)
    node = model_out()
    rclpy.spin(node)
    node.trt_engine.close()
    node.destroy_node()
    rclpy.shutdown()
