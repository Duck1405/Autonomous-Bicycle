import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import pygame


class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        pygame.init()
        self.screen = pygame.display.set_mode((420, 180))
        pygame.display.set_caption("ROS 2 Keyboard Control")

        self.linear_speed = 1.0
        self.angular_speed = 1.5
        self.last_command = None

        self.timer = self.create_timer(0.05, self.update)

        self.get_logger().info("Keyboard control started. Click the pygame window, then use W/A/S/D.")

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                return

        self.screen.fill((20, 20, 20))
        pygame.display.flip()

        keys = pygame.key.get_pressed()

        msg = Twist()
        command = "stop"

        # Forward/backward
        if keys[pygame.K_w]:
            msg.linear.x = self.linear_speed
            command = "forward"

        if keys[pygame.K_s]:
            msg.linear.x = -self.linear_speed
            command = "backward"

        # Left/right steering
        if keys[pygame.K_a]:
            msg.angular.z = self.angular_speed
            command = "left"

        if keys[pygame.K_d]:
            msg.angular.z = -self.angular_speed
            command = "right"

        # Emergency stop
        if keys[pygame.K_SPACE]:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            command = "emergency stop"

        if command != self.last_command:
            self.get_logger().info(
                f"Command: {command} "
                f"(linear.x={msg.linear.x:.2f}, angular.z={msg.angular.z:.2f})"
            )
            self.last_command = command

        self.publisher_.publish(msg)

    def destroy_node(self):
        pygame.quit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = KeyboardControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
