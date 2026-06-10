import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import Twist
from pynput import keyboard


class Keyboard(Node):
    def __init__(self):
        super().__init__('keyboard_output_node')
        self.get_logger().info('Keyboard Output node has been started.')
        self.subscription = self.create_subscription(Twist, 'keyboard_input', self.listener_callback, 10)
    def listener_callback(self, msg):
        self.get_logger().info(f"Received Twist message: linear.x={msg.linear.x}, angular.z={msg.angular.z}")
        

def main(args=None):
    rclpy.init(args=args)
    keyboard_output_node = Keyboard()
    try:
        rclpy.spin(keyboard_output_node)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard_output_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()