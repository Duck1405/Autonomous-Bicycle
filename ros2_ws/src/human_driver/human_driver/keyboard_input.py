import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import Twist
from pynput import keyboard

class Keyboard(Node):
    def __init__(self):
        super().__init__('keyboard_node')
        self.get_logger().info('Keyboard node has been started.')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)  
        self.listener = keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release)
        self.listener.start()
        self.get_logger().info("Keyboard Teleop Started. Use W, A, S, D to move. Space to stop.")
        
    def on_press(self, key):
        twist = Twist()
        try:
            if key.char == 'w':
                twist.linear.x = 1.0
            if key.char == 's':
                twist.linear.x = -1.0
            if key.char == 'a':
                twist.angular.z = 1.0
            if key.char == 'd':
                twist.angular.z = -1.0
        except AttributeError:
            pass
        
        self.publisher_.publish(twist)
        
    def on_release(self, key):
        self.get_logger().info(f"Key released: {key}")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)
        
def main(args=None):
    rclpy.init(args=args)
    teleop_node = Keyboard()
    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        pass
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        