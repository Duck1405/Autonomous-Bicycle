import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class LaneVisualizerNode(Node):
    def __init__(self):
        super().__init__('lane_visualizer_node')
        
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # State storage for incoming lane data
        self.left_lane_pts = None
        self.right_lane_pts = None

        # 1. Subscribe to the raw left camera feed
        self.image_sub = self.create_subscription(
            Image, '/stereo/left/image_raw', self.image_callback, 10
        )

        # 2. Subscribe to Left and Right Lanes from LaneATT
        self.left_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/left_lane', self.left_lane_callback, 10
        )
        self.right_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/right_lane', self.right_lane_callback, 10
        )

        self.get_logger().info("Lane Visualizer node started (Raw Image + LaneATT).")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

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

        # Trigger rendering whenever new right lane data arrives (keeps cadence synchronized)
        self.render_and_display()

    def render_and_display(self):
        if self.latest_frame is None:
            return

        vis_frame = self.latest_frame.copy()
        h, w, _ = vis_frame.shape

        # --- Draw Lane Lines ---
        # LaneATT outputs normalized coordinates (0.0 to 1.0), scale them back to frame pixels (w, h)
        # Left Lane = Blue, Right Lane = Red
        for lane_pts, color in [(self.left_lane_pts, (255, 0, 0)), (self.right_lane_pts, (0, 0, 255))]:
            if lane_pts is not None and len(lane_pts) > 1:
                pixel_pts = []
                for pt in lane_pts:
                    px = int(pt[0] * w)
                    py = int(pt[1] * h)
                    pixel_pts.append((px, py))
                
                # Draw lines connecting sequential points of the lane polygon
                for i in range(len(pixel_pts) - 1):
                    cv2.line(vis_frame, pixel_pts[i], pixel_pts[i+1], color, 4)

        # --- Show Debug Window ---
        cv2.imshow("LaneATT Debug - Raw + Lanes", vis_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
