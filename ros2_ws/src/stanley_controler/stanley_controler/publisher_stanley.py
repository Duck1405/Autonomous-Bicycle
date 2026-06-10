import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import math


class StanleyPublisher(Node):
    def __init__(self):
        super().__init__('stanley_controler_node')
        
        self.k = 1.2
        self.softening = 1.0
        self.target_speed = 2.0
        self.max_angular_z = 1.5

        self.heading_error = None
        self.cross_track_error = None
        self.speed = 0.0

        self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # self.create_subscription(
        #     Float64,
        #     '/heading_error',
        #     self.heading_error_callback,
        #     10
        # )
        # self.create_subscription(
        #     Float64,
        #     '/cross_track_error',
        #     self.cross_track_error_callback,
        #     10
        # )
        # self.timer = self.create_timer(0.02, self.control_loop)
    
    def control_loop(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.sqrt(vx * vx + vy * vy)
        
        
    def odom_callback(self, msg):
        # Extract the necessary information from the Odometry message
        # For simplicity, let's assume we are just publishing a constant velocity command
        twist = Twist()
        twist.linear.x = 1.0  # Set a constant forward velocity
        twist.angular.z = 0.0  # No angular velocity for now
        
        self.get_logger().info(f"Published Twist message: linear.x={twist.linear.x}, angular.z={twist.angular.z}")
    
    def heading_error_callback(self, msg):
        # Here you would implement the logic to adjust the angular velocity based on the heading error
        # For demonstration, let's just log the heading error
        self.get_logger().info(f"Received heading error: {msg.data}")
        
    def cross_track_error_callback(self, msg):
        # Here you would implement the logic to adjust the linear velocity based on the cross-track error
        # For demonstration, let's just log the cross-track error
        self.get_logger().info(f"Received cross-track error: {msg.data}")
    
def main(args=None):
    rclpy.init(args=args)
    stanley_publisher = StanleyPublisher()
    rclpy.spin(stanley_publisher)
    stanley_publisher.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()

