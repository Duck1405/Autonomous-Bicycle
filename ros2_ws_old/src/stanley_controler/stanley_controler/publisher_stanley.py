import csv
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from pathlib import Path


class StanleyPublisher(Node):
    def __init__(self):
        super().__init__('stanley_controler_node')

        self.k = 1.2
        self.softening = 1.0
        self.target_speed = 8.0
        self.max_angular_z = 1.5
        self.front_axle_offset = 2

        self.heading_error = None
        self.cross_track_error = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.odom_count = 0
        self.last_target_index = 0
        self.path = self.load_centerline()

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.get_logger().info(f"Loaded {len(self.path)} centerline points.")

    def load_centerline(self):
        csv_path = Path(__file__).with_name('centerline.csv')
        points = []

        with csv_path.open(newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                points.append((float(row['x']), float(row['y'])))

        if len(points) < 2:
            raise RuntimeError(f"Need at least 2 centerline points in {csv_path}")

        path = []
        for index, (x, y) in enumerate(points):
            if index < len(points) - 1:
                next_x, next_y = points[index + 1]
                yaw = math.atan2(next_y - y, next_x - x)
            else:
                prev_x, prev_y = points[index - 1]
                yaw = math.atan2(y - prev_y, x - prev_x)

            path.append((x, y, yaw))

        return path

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist.linear

        self.x = position.x
        self.y = position.y
        self.yaw = self.quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.speed = math.hypot(velocity.x, velocity.y)

        angular_z, target_index = self.compute_stanley_command()
        self.last_target_index = target_index

        twist = Twist()
        twist.linear.x = self.target_speed
        twist.angular.z = angular_z

        self.publisher_.publish(twist)
        self.odom_count += 1
        if self.odom_count % 20 == 0:
            self.get_logger().info(
                f"car pose: x={self.x:.2f}, y={self.y:.2f}, "
                f"yaw={self.yaw:.2f} rad, speed={self.speed:.2f} m/s, "
                f"target={target_index}, heading_error={self.heading_error:.2f}, "
                f"cross_track_error={self.cross_track_error:.2f}, "
                f"angular_z={angular_z:.2f}"
            )

    def compute_stanley_command(self):
        front_x = self.x + self.front_axle_offset * math.cos(self.yaw)
        front_y = self.y + self.front_axle_offset * math.sin(self.yaw)

        target_index = self.find_nearest_path_index(front_x, front_y)
        target_x, target_y, target_yaw = self.path[target_index]

        dx = front_x - target_x
        dy = front_y - target_y
        path_left_normal_error = (
            math.cos(target_yaw) * dy
            - math.sin(target_yaw) * dx
        )

        self.cross_track_error = path_left_normal_error
        self.heading_error = self.normalize_angle(target_yaw - self.yaw)

        crosstrack_correction = math.atan2(
            self.k * self.cross_track_error,
            self.speed + self.softening,
        )

        angular_z = self.heading_error - crosstrack_correction
        angular_z = self.clamp(angular_z, -self.max_angular_z, self.max_angular_z)

        return angular_z, target_index

    def find_nearest_path_index(self, x, y):
        search_start = max(0, self.last_target_index - 5)
        search_end = min(len(self.path), self.last_target_index + 30)

        if search_start >= search_end:
            search_start = 0
            search_end = len(self.path)

        best_index = search_start
        best_distance = float('inf')

        for index in range(search_start, search_end):
            path_x, path_y, _ = self.path[index]
            distance = (x - path_x) ** 2 + (y - path_y) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))


def main(args=None):
    rclpy.init(args=args)
    stanley_publisher = StanleyPublisher()
    rclpy.spin(stanley_publisher)
    stanley_publisher.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
