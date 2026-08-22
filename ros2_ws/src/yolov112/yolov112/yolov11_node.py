import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

from .trt_runner import CudaRT, TrtEngine

class Yolov11Node(Node):
    def __init__(self):
        super().__init__('yolov112_node')
        self.engine = "/home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/YoloN/YoloN_fb16.engine"
        self.warmup = 20
        self.CudaRT = CudaRT()
        self.trt_engine = TrtEngine(self.engine, self.CudaRT)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/stereo/depth/color', self.image_callback, 10)  # Ros2: "/left/raw", "/right/raw"
        # raw_camera left: /dev/video0, right: /dev/video1
        self.det_pub = self.create_publisher(Float32MultiArray, '/yolov11/detections', 10)

        for _ in range(self.warmup):
            self.trt_engine.run(self.pre_yolo(np.zeros((360, 640, 3), dtype=np.uint8)))

        self.get_logger().info('Yolov11 node started')
    
    def yolo_decode(self, raw, r, dx, dy, conf_threshold=0.35):
        """(1,300,6) engine output -> [(p1, p2, conf, cls)] in original frame coords.

        NMS is already inside the graph; rows are [x1,y1,x2,y2,conf,cls] in
        letterboxed 640-space and padding rows have conf == 0.
        """
        dets = np.asarray(raw, dtype=np.float32).reshape(-1, 6)
        out = []
        for x1, y1, x2, y2, conf, cls in dets[dets[:, 4] >= conf_threshold]:
            p1 = (int((x1 - dx) / r), int((y1 - dy) / r))
            p2 = (int((x2 - dx) / r), int((y2 - dy) / r))
            out.append((p1, p2, float(conf), int(cls)))
        return out

    def pre_yolo_meta(self, frame, size=640):
        """BGR frame -> (1,3,size,size) /255, RGB order, letterboxed to square."""
        h, w = frame.shape[:2]
        r = min(size / h, size / w)
        nw, nh = round(w * r), round(h * r)
        canvas = np.full((size, size, 3), 114, np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob, dtype=np.float32) / 255.0, r, dx, dy

    def pre_yolo(self, frame, size=640):
        return self.pre_yolo_meta(frame, size)[0]

    def get_inference(self, frame):
        """Run YoloV11 on one BGR frame, return the decoded detections.

        output0 (1,300,6) engine output -> [(p1, p2, conf, cls)] in original
        frame pixel coords; NMS is already inside the graph (postprocess.yolo_decode).
        """
        arr, r, dx, dy = self.pre_yolo_meta(frame)
        outputs = self.trt_engine.infer(arr)
        raw = next(iter(outputs.values()))
        return self.yolo_decode(raw, r, dx, dy)

    def rg10_to_bgr(self, msg):
        """RG10 (V4L2 SRGGB10, 10-bit Bayer RGGB padded into 16-bit words) -> BGR8."""
        dtype = np.dtype(np.uint16).newbyteorder('>' if msg.is_bigendian else '<')
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)
        raw = raw[:, :msg.width]
        raw8 = (raw >> 2).astype(np.uint8)          # 10-bit (0-1023) -> 8-bit (0-255)
        return cv2.cvtColor(raw8, cv2.COLOR_BayerRG2BGR)

    @staticmethod
    def _to_msg(objects):
        """Detections -> Float32MultiArray, rows of [x1,y1,x2,y2,conf,cls] flattened."""
        msg = Float32MultiArray()
        rows = [[*p1, *p2, conf, float(cls)] for p1, p2, conf, cls in objects]
        msg.data = np.asarray(rows, dtype=np.float32).flatten().tolist() if rows else []
        return msg

    def image_callback(self, frame):
    # Your stereo camera appears to publish RG10 Bayer images
        t0 = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')
        t1 = time.perf_counter()
        self.get_logger().info(f'Image conversion took {t1 - t0:.4f} seconds. FPS: {1/(t1 - t0):.2f}')
        self.get_logger().info(f'Image received, type: {type(frame)}')
        
        t2 = time.perf_counter()
        objects = self.get_inference(frame)
        t3 = time.perf_counter()
        self.get_logger().info(f'Image Inference took {t1 - t0:.4f} seconds')
        
        self.get_logger().info(f'Detected {len(objects)} objects')

        for i, (p1, p2, conf, cls) in enumerate(objects, start=1):
            cx = (p1[0] + p2[0]) // 2
            cy = (p1[1] + p2[1]) // 2
            self.get_logger().info(
                f'  Object {i}: class={cls}, confidence={conf:.2f}, '
                f'box=({p1[0]}, {p1[1]}) to ({p2[0]}, {p2[1]}), '
                f'center=({cx}, {cy})'
            )

        self.get_logger().info(f"Total Time: {t3 - t0:.4f} seconds. FPS: {1/(t3 - t0):.2f}")   
        self.det_pub.publish(self._to_msg(objects))


def main(args=None):
    rclpy.init(args=args)
    node = Yolov11Node()
    rclpy.spin(node)
    node.trt_engine.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
