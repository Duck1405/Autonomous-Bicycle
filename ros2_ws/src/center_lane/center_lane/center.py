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
        self.mid_lane_pub = self.create_publisher(Float32MultiArray, "/laneatt/mid_lane", 1)

        self.left_lane_pts = None
        self.right_lane_pts = None
        # Per-y-row EMA of ego-lane width in pixels, learned while both edges are
        # visible. Width varies with y (perspective), so it must be per-row, not
        # a single scalar. Used to synthesize a missing edge from the visible one.
        self.lane_width_by_y = {}
        self.mid_points = None

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

        self.check_both()

    def check_both(self):
        if self.right_lane_pts is not None and self.left_lane_pts is not None:
            self.get_logger().info("Both Found")
            self.get_center()
        elif self.right_lane_pts == None:
            self.get_logger().info("Right Lane Not Found")
        else:
            self.get_logger().info("Left Lane Not Found")

    def get_center_two(self):
        pass

    def get_center(self):
        left_points = self.left_lane_pts
        right_points = self.right_lane_pts

        # `synthesized` names the edge ('left'/'right') that was inferred from
        # the width prior instead of detected, or None when both edges are real.
        synthesized = None
        if left_points is not None and right_points is not None:
            # Both edges visible: learn the per-row lane width (EMA, alpha=0.2).
            left_by_y = {int(y): x for x, y in left_points}
            right_by_y = {int(y): x for x, y in right_points}
            for y in set(left_by_y) & set(right_by_y):
                w = right_by_y[y] - left_by_y[y]
                if w <= 0:
                    continue
                old = self.lane_width_by_y.get(y)
                self.lane_width_by_y[y] = w if old is None else 0.8 * old + 0.2 * w
        elif self.lane_width_by_y:
            # One edge missing: synthesize it by offsetting the visible edge by
            # the learned width, at rows where both a point and a width exist.
            visible = left_points if left_points is not None else right_points
            sign = 1 if left_points is not None else -1   # left visible -> right = x + w
            synth = [[x + sign * self.lane_width_by_y[int(y)], y]
                     for x, y in visible if int(y) in self.lane_width_by_y]
            if len(synth) < 2:
                self.mid_points = None
                self._publish_mid_lane(None)
                return None   # too little prior overlap to trust
            synth = np.array(synth).round().astype(int)
            if left_points is not None:
                right_points, synthesized = synth, 'right'
            else:
                left_points, synthesized = synth, 'left'
        else:
            self.mid_points = None
            self._publish_mid_lane(None)
            return None

        # Midpoints between the two ego lanes, one per shared y-row.
        left_by_y = {int(y): x for x, y in left_points}
        right_by_y = {int(y): x for x, y in right_points}
        shared_ys = sorted(set(left_by_y) & set(right_by_y))
        mid_points = np.array([[(left_by_y[y] + right_by_y[y]) / 2, y] for y in shared_ys], dtype=int)

        self.mid_points = mid_points
        self._publish_mid_lane(mid_points)
        self.get_logger().info(
            f"Center lane: {len(mid_points)} midpoints"
            + (f" (synthesized {synthesized} edge)" if synthesized else "")
        )
        return mid_points

    def _publish_mid_lane(self, mid_points):
        """Publish on /laneatt/mid_lane in the same flat [x, y, x, y, ...]
        Float32MultiArray layout as /laneatt/left_lane and /laneatt/right_lane."""
        msg = Float32MultiArray()
        if mid_points is not None and len(mid_points) > 0:
            msg.data = np.asarray(mid_points, dtype=np.float32).ravel().tolist()
        else:
            msg.data = []
        self.mid_lane_pub.publish(msg)

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
