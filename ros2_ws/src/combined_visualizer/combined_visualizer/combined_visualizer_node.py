import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

class CombinedVisualizerNode(Node):
    def __init__(self):
        super().__init__('combined_visualizer_node')
        
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # State tracking for all data streams
        self.latest_detections = []
        self.left_lane_pts = None
        self.right_lane_pts = None

        # For FPS calculation
        self.last_time = time.perf_counter()

        # 1. Subscribe to the raw left camera feed
        self.image_sub = self.create_subscription(
            Image, '/stereo/left/image_raw', self.image_callback, 10
        )

        # 2. Subscribe to YOLO detections
        self.detection_sub = self.create_subscription(
            Float32MultiArray, '/yolov11/detections', self.detection_callback, 10
        )

        # 3. Subscribe to LaneATT left & right lanes
        self.left_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/left_lane', self.left_lane_callback, 10
        )
        self.right_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/right_lane', self.right_lane_callback, 10
        )

        self.get_logger().info("Combined Visualizer Node started (Raw + YOLO + LaneATT with FPS).")

    def image_callback(self, msg):
        try:
            # Calculate loop FPS based on incoming frame frequency
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            if dt > 0:
                fps = 1.0 / dt
                self.get_logger().info(f"Visualizer Loop FPS: {fps:.2f}")

            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.render_and_display()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def detection_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.latest_detections = data.reshape(-1, 6)
        else:
            self.latest_detections = []

    def left_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.left_lane_pts = data.reshape(-1, 2)
        else:
            self.left_lane_pts = None

    def right_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.right_lane_pts = data.reshape(-1, 2)
        else:
            self.right_lane_pts = None

    def render_and_display(self):
        if self.latest_frame is None:
            return

        vis_frame = self.latest_frame.copy()
        h, w, _ = vis_frame.shape

        # --- A. Draw Lane Lines (LaneATT) ---
        # Left lane = Blue, Right lane = Red (Normalized [0,1] scaled to image pixels)
        for lane_pts, color in [(self.left_lane_pts, (255, 0, 0)), (self.right_lane_pts, (0, 0, 255))]:
            if lane_pts is not None and len(lane_pts) > 1:
                pixel_pts = []
                for pt in lane_pts:
                    px = int(pt[0] * w)
                    py = int(pt[1] * h)
                    pixel_pts.append((px, py))
                
                for i in range(len(pixel_pts) - 1):
                    cv2.line(vis_frame, pixel_pts[i], pixel_pts[i+1], color, 4)

        # --- B. Draw YOLO Bounding Boxes ---
        for det in self.latest_detections:
            x1, y1, x2, y2, conf, cls = det
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cls_id = int(cls)

            # Box = Green
            cv2.rectangle(vis_frame, p1, p2, (0, 255, 0), 2)
            
            # Label
            label = f"C{cls_id}: {conf:.2f}"
            cv2.putText(vis_frame, label, (p1[0], max(p1[1] - 10, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- C. Display Final Window ---
        cv2.imshow("Autonomous Bicycle - Combined Debug View", vis_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CombinedVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
