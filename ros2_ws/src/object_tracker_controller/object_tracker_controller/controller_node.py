import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int8
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ObjectTrackerController(Node):
    def __init__(self):
        super().__init__('object_tracker_controller')
        
        self.bridge = CvBridge()
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.latest_detections = []
        
        # Lane state tracking for the visual overlay
        self.left_lane_pts = None
        self.right_lane_pts = None

        # 1. Subscriptions
        self.rgb_sub = self.create_subscription(
            Image, '/stereo/left/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/stereo/depth', self.depth_callback, 10
        )
        self.det_sub = self.create_subscription(
            Float32MultiArray, '/yolov11/detections', self.detection_callback, 10
        )
        self.left_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/left_lane', self.left_lane_callback, 10
        )
        self.right_lane_sub = self.create_subscription(
            Float32MultiArray, '/laneatt/right_lane', self.right_lane_callback, 10
        )

        # 2. Publisher for the speed command (+1, 0, -1)
        self.speed_cmd_pub = self.create_publisher(Int8, '/bicycle/speed_command', 10)

        self.get_logger().info("Object Tracker Controller & Dual Visualizer Node Initialized.")

    def rgb_callback(self, msg):
        try:
            self.latest_rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame_and_control()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            # Assuming 32FC1 meters format
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Depth Error: {e}")

    def detection_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.latest_detections = data.reshape(-1, 6)
        else:
            self.latest_detections = []

    def left_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.left_lane_pts = data.reshape(-1, 2) if len(data) > 0 else None

    def right_lane_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.right_lane_pts = data.reshape(-1, 2) if len(data) > 0 else None

    def process_frame_and_control(self):
        if self.latest_rgb_frame is None:
            return

        target_command = 1  # Default: Speed up (+1) if path is clear
        rgb_vis = self.latest_rgb_frame.copy()
        h, w, _ = rgb_vis.shape

        # --- A. Draw Lanes on RGB View ---
        for lane_pts, color in [(self.left_lane_pts, (255, 0, 0)), (self.right_lane_pts, (0, 0, 255))]:
            if lane_pts is not None and len(lane_pts) > 1:
                pixel_pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in lane_pts]
                for i in range(len(pixel_pts) - 1):
                    cv2.line(rgb_vis, pixel_pts[i], pixel_pts[i+1], color, 4)

        # --- B. Process YOLO & Depth Logic ---
        best_obj_distance = None
        if len(self.latest_detections) > 0:
            valid_objects = []
            for det in self.latest_detections:
                x1, y1, x2, y2, conf, cls = det
                cls_id = int(cls)
                if cls_id in [0, 1]:  # 0: person, 1: car
                    valid_objects.append((x1, y1, x2, y2, cls_id, conf))

            if valid_objects:
                frame_center_x = w / 2.0
                best_obj = min(valid_objects, key=lambda obj: abs(((obj[0] + obj[2]) / 2.0) - frame_center_x))
                x1, y1, x2, y2, cls_id, conf = best_obj

                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                distance_str = "N/A"

                if self.latest_depth_frame is not None:
                    dh, dw = self.latest_depth_frame.shape[:2]
                    dx1, dx2 = max(0, int(x1)), min(dw, int(x2))
                    dy1, dy2 = max(0, int(y1)), min(dh, int(y2))
                    box_roi = self.latest_depth_frame[dy1:dy2, dx1:dx2]

                    if box_roi.size > 0:
                        valid_depths = box_roi[np.isfinite(box_roi) & (box_roi > 0)]
                        if valid_depths.size > 0:
                            avg_distance_meters = np.median(valid_depths)
                            distance_feet = avg_distance_meters * 3.28084
                            best_obj_distance = distance_feet
                            distance_str = f"{distance_feet:.1f} ft"

                # Draw YOLO Box and Distance on RGB View
                cv2.rectangle(rgb_vis, p1, p2, (0, 255, 0), 2)
                label = f"C{cls_id} {conf:.2f} | {distance_str}"
                cv2.putText(rgb_vis, label, (p1[0], max(p1[1] - 10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- C. Evaluate Control Rules ---
        if best_obj_distance is not None:
            if 8.0 <= best_obj_distance <= 10.0:
                target_command = 0
            elif best_obj_distance < 8.0:
                target_command = -1
            else:
                target_command = 1
        else:
            target_command = 1  # No valid target, clear path to speed up

        # Publish Speed Command
        cmd_msg = Int8()
        cmd_msg.data = target_command
        self.speed_cmd_pub.publish(cmd_msg)

        # --- D. Show RGB Window ---
        cv2.imshow("Controller - RGB Combined View", rgb_vis)

        # --- E. Show Depth Map Window ---
        if self.latest_depth_frame is not None:
            depth_normalized = cv2.normalize(self.latest_depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            for det in self.latest_detections:
                x1, y1, x2, y2, _, cls = det
                if int(cls) in [0, 1]:
                    cv2.rectangle(depth_color, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            cv2.imshow("Controller - Depth Map Debug View", depth_color)

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerController()
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

