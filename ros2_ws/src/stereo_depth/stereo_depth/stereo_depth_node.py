import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import vpi


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.bridge = CvBridge()

        # Camera specs from Waveshare IMX219-83 datasheet
        self.baseline_m = 0.060       # 60mm
        self.focal_length_mm = 2.6
        self.sensor_width_mm = 3.68   # 1/4" sensor typical width, approx for IMX219
        self.image_width_px = 640     # matches the downsampled width used below

        self.focal_px = (self.focal_length_mm / self.sensor_width_mm) * self.image_width_px

        self.max_disparity = 32  # halved to match halved resolution

        left_sub = message_filters.Subscriber(self, Image, '/left/image_raw')
        right_sub = message_filters.Subscriber(self, Image, '/right/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.stereo_callback)

        self.depth_pub = self.create_publisher(Image, '/stereo/depth_image', 10)
        self.disparity_pub = self.create_publisher(Image, '/stereo/disparity_viz', 10)

        self.get_logger().info('Stereo depth node started (VPI CUDA+PVA backend)')

    def stereo_callback(self, left_msg, right_msg):
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Downsample before matching -- biggest single speed lever
        left_gray = cv2.resize(left_gray, (640, 360))
        right_gray = cv2.resize(right_gray, (640, 360))

        with vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left_gray).convert(vpi.Format.Y16_ER)
            right_vpi = vpi.asimage(right_gray).convert(vpi.Format.Y16_ER)

        with vpi.Backend.CUDA:
            disparity_vpi = vpi.stereodisp(
                left_vpi, right_vpi, window=5, maxdisp=self.max_disparity)

        with vpi.Backend.CUDA:
            disparity_u8 = disparity_vpi.convert(
                vpi.Format.U8, scale=255.0 / (32 * self.max_disparity))

        with disparity_vpi.rlock_cpu() as disp_data:
            # VPI encodes disparity in Q10.5 fixed point -- divide by 32 to get real disparity
            disparity_raw = np.array(disp_data).astype(np.float32) / 32.0

        with np.errstate(divide='ignore'):
            depth_m = (self.focal_px * self.baseline_m) / disparity_raw
        depth_m[disparity_raw <= 0] = 0

        depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding='32FC1')
        depth_msg.header = left_msg.header
        self.depth_pub.publish(depth_msg)

        with disparity_u8.rlock_cpu() as viz_data:
            viz_np = np.array(viz_data)
        viz_msg = self.bridge.cv2_to_imgmsg(viz_np, encoding='mono8')
        viz_msg.header = left_msg.header
        self.disparity_pub.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
