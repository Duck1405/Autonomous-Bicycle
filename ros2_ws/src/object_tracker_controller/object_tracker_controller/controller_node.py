import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int8
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ObjectTrackerController(Node):
    def __init__(self):
        super().__init__('object_tracker_controller')
        
        self.bridge = CvBridge()
        self.latest_depth_frame = None
        self.latest_detections = []

        # 1. Subscriptions
        # Sub to depth image (usually 32FC1 or 16UC1 depending on how stereolabs/depth_image_proc publishes it)
        self.depth_sub = self.create_subscription(
            Image, '/stereo/depth', self.depth_callback, 10
        )
        # Sub to your YOLO detections
        self.det_sub = self.create_subscription(
            Float32MultiArray, '/yolov11/detections', self.detection_callback, 10
        )

        # 2. Publisher for the speed command (+1, 0, -1)
        self.speed_cmd_pub = self.create_publisher(Int8, '/bicycle/speed_command', 10)

        self.get_logger().info("Object Tracker & Speed Controller Node Initialized.")

    def depth_callback(self, msg):
        try:
            # Depth maps are typically floating-point meters (32FC1) or millimeters (16UC1). 
            # Assuming 32FC1 meters here; adjust encoding if your stereo node uses 16UC1.
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Depth Error: {e}")

    def detection_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.latest_detections = data.reshape(-1, 6)
        else:
            self.latest_detections = []

        # Process control logic whenever new detections arrive
        self.evaluate_control_logic()

    def evaluate_control_logic(self):
        # Default behavior if no objects are detected or depth is missing
        target_command = 1  # Default: Speed up (+1) if path is clear

        if self.latest_depth_frame is not None and len(self.latest_detections) > 0:
            h_depth, w_depth = self.latest_depth_frame.shape[:2]
            
            # We are interested in Class 0 (person) or Class 1 (car)
            valid_objects = []
            for det in self.latest_detections:
                x1, y1, x2, y2, conf, cls = det
                cls_id = int(cls)
                if cls_id in [0, 1]:  # 0: person, 1: car
                    valid_objects.append((x1, y1, x2, y2, cls_id, conf))

            if valid_objects:
                # Find the object closest to the center of the frame or closest distance
                # For simplicity, let's look for the object with the largest bounding box area (closest) 
                # or closest center alignment. Let's pick the one closest to the center X axis.
                frame_center_x = w_depth / 2.0
                
                best_obj = min(valid_objects, key=lambda obj: abs(((obj[0] + obj[2]) / 2.0) - frame_center_x))
                x1, y1, x2, y2, cls_id, conf = best_obj

                # Clamp box coordinates to depth frame dimensions
                x1, x2 = max(0, int(x1)), min(w_depth, int(x2))
                y1, y2 = max(0, int(y1)), min(h_depth, int(y2))

                # Extract the bounding box region from the depth map
                box_depth_roi = self.latest_depth_frame[y1:y2, x1:x2]

                if box_depth_roi.size > 0:
                    # Filter out NaN, infinite, or zero depth values (invalid pixels)
                    valid_depths = box_depth_roi[np.isfinite(box_depth_roi) & (box_depth_roi > 0)]

                    if valid_depths.size > 0:
                        avg_distance_meters = np.median(valid_depths)  # Median is more robust to outliers than mean
                        
                        # Convert meters to feet (1 meter = 3.28084 feet)
                        distance_feet = avg_distance_meters * 3.28084

                        self.get_logger().info(
                            f"Target Detected [Class {cls_id}] | Avg Distance: {distance_feet:.2f} ft"
                        )

                        # --- Speed Control Logic ---
                        # Target range: 8 to 10 feet -> Maintain Speed (0)
                        # Too close (< 8 feet) -> Slow Down (-1)
                        # Too far (> 10 feet) -> Speed Up (+1)
                        if 8.0 <= distance_feet <= 10.0:
                            target_command = 0
                            self.get_logger().info("Target in range (8-10 ft). Maintaining speed (0).")
                        elif distance_feet < 8.0:
                            target_command = -1
                            self.get_logger().warn("Obstacle too close (< 8 ft)! Slowing down (-1).")
                        else:
                            target_command = 1
                            self.get_logger().info("Target far (> 10 ft). Speeding up (+1).")

        # Publish the discrete speed command
        cmd_msg = Int8()
        cmd_msg.data = target_command
        self.speed_cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
