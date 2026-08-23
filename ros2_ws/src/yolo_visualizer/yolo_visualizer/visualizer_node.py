import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class YoloVisualizerNode(Node):
    def __init__(self):
        super().__init__('yolo_visualizer_node')
        
        self.bridge = CvBridge()
        self.latest_frame = None

        # 1. Subscribe to the raw left camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/stereo/left/image_raw',
            self.image_callback,
            10
        )

        # 2. Subscribe to your YOLO detections Float32MultiArray topic
        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            '/yolov11/detections',
            self.detection_callback,
            10
        )

        # Store the latest parsed detections so they persist smoothly across frames
        self.latest_detections = []

        self.get_logger().info("YOLO Visualizer node started, listening for image and detection topics...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV BGR format
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def detection_callback(self, msg):
        # Unpack the flat Float32MultiArray back into rows of 6 elements: [x1, y1, x2, y2, conf, cls]
        data = np.array(msg.data, dtype=np.float32)
        if len(data) > 0:
            self.latest_detections = data.reshape(-1, 6)
        else:
            self.latest_detections = []

        # If we have a frame, draw the latest available detections onto it
        if self.latest_frame is not None:
            vis_frame = self.latest_frame.copy()

            for det in self.latest_detections:
                x1, y1, x2, y2, conf, cls = det
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cls_id = int(cls)

                # Draw bounding box
                cv2.rectangle(vis_frame, p1, p2, (0, 255, 0), 2)
                
                # Draw label text
                label = f"Class {cls_id}: {conf:.2f}"
                cv2.putText(vis_frame, label, (p1[0], max(p1[1] - 10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the overlapping visual
            cv2.imshow("Stereo Left - YOLO Overlays", vis_frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisualizerNode()
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
