import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

from .trt_runner import CudaRT, TrtEngine
from .postprocess import laneatt_decode


class LaneATTNode(Node):
    def __init__(self):
        super().__init__('laneAtt_node')
        self.engine = "/home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine"
        self.warmup = 50
        self.CudaRT = CudaRT()
        self.trt_engine = TrtEngine(self.engine, self.CudaRT)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/dev/video0', self.image_callback, 10)  # Ros2: "/left/raw", "/right/raw"
        # raw_camera left: /dev/video0, right: /dev/video1
        self.left_pub = self.create_publisher(Float32MultiArray, '/laneatt/left_lane', 10)
        self.right_pub = self.create_publisher(Float32MultiArray, '/laneatt/right_lane', 10)

        for _ in range(self.warmup):
            self.trt_engine.run(self.pre_laneatt(np.zeros((360, 640, 3), dtype=np.uint8)))

        self.get_logger().info('LaneATT node started')

    def pre_laneatt(self, frame):
        """BGR frame -> (1,3,360,640) /255, BGR order kept (mirrors LaneATT.frame_eval)."""
        img = cv2.resize(frame, (640, 360))
        arr = img.astype(np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    def get_inference(self, frame):
        """Run LaneATT on one BGR frame, return the decoded lane list.

        proposals[1,1000,77] engine output -> up to 2 lane dicts (laneatt_decode's
        default nms_topk), each {"points": (N,2) normalized (x,y), "conf", ...}.
        """
        arr = self.pre_laneatt(frame)
        outputs = self.trt_engine.infer(arr)
        raw = next(iter(outputs.values()))
        return laneatt_decode(raw)

    @staticmethod
    def _bottom_x(lane):
        """Normalized x where a lane's polyline is closest to the camera (largest y)."""
        pts = lane["points"]
        return float(pts[np.argmax(pts[:, 1]), 0])

    def split_left_right(self, lanes):
        """lane dicts -> (left, right), ordered by bottom-row x position.

        laneatt_decode keeps at most 2 lanes by default (nms_topk=2), which for
        ego-lane detection are the left and right boundary. Missing side(s) come
        back as None rather than guessing.
        """
        if not lanes:
            return None, None
        scored = sorted(lanes, key=self._bottom_x)
        if len(scored) == 1:
            return (scored[0], None) if self._bottom_x(scored[0]) < 0.5 else (None, scored[0])
        return scored[0], scored[-1]

    @staticmethod
    def _to_msg(lane):
        msg = Float32MultiArray()
        if lane is not None:
            msg.data = lane["points"].astype(np.float32).flatten().tolist()
        return msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info('Image received, shape: {}'.format(frame.shape))
        lanes = self.get_inference(frame)
        self.get_logger().info(f"Detected {len(lanes)} lanes")
        left, right = self.split_left_right(lanes)
        
        if left is not None:
            self.get_logger().info(f"left points shape: {left['points'].shape}")
        if right is not None:
            self.get_logger().info(f"right points shape: {right['points'].shape}")
        self.left_pub.publish(self._to_msg(left))
        self.right_pub.publish(self._to_msg(right))


def main(args=None):
    rclpy.init(args=args)
    node = LaneATTNode()
    rclpy.spin(node)
    node.trt_engine.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
