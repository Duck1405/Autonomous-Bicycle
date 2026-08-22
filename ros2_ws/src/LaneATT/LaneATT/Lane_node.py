import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

from .trt_runner import CudaRT, TrtEngine

class LaneATTNode(Node):
    def __init__(self):
        super().__init__('laneAtt_node')
        self.engine = "/home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine"
        self.warmup = 50
        self.CudaRT = CudaRT()
        self.trt_engine = TrtEngine(self.engine, self.CudaRT)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/stereo/left/image_raw', self.image_callback, 10)  # Ros2: "/left/raw", "/right/raw"
        # raw_camera left: /dev/video0, right: /dev/video1
        self.left_pub = self.create_publisher(Float32MultiArray, '/laneatt/left_lane', 10)
        self.right_pub = self.create_publisher(Float32MultiArray, '/laneatt/right_lane', 10)

        for _ in range(self.warmup):
            self.trt_engine.run(self.pre_laneatt(np.zeros((360, 640, 3), dtype=np.uint8)))

        self.get_logger().info('LaneATT node started')
    
    
    def _softmax2(self, logits):
        """Row-wise softmax over a (N, 2) block, max-subtracted for stability."""
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def lane_nms(self, proposals, scores, overlap=50.0, top_k=2):
        """Greedy lane NMS. Returns kept indices into `proposals`, best score first.

        "Overlap" is the mean absolute horizontal distance between two lanes over the
        vertical span where both exist; the lower-scored lane is suppressed when that
        mean is below `overlap` (in input pixels).
        """
        n = proposals.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)

        n_offsets = proposals.shape[1] - 5
        n_strips = n_offsets - 1
        order = np.argsort(-scores, kind="stable")

        # A non-positive threshold can never suppress (mean distance is >= 0 and the
        # test is strict), so short-circuit the O(N^2) loop. Matches the fast path in
        # lib/nms_pytorch.py.
        if overlap <= 0:
            return order[:min(n, int(top_k))]

        boxes = proposals[order]
        # Vertical extent in offset-index space, matching devIoU in the CUDA kernel.
        starts = (boxes[:, 2] * n_strips + 0.5).astype(np.int64)
        lengths = boxes[:, 4]
        ends = ((starts.astype(np.float64) + lengths - 1 + 0.5).astype(np.int64)
                - (lengths - 1 < 0).astype(np.int64))
        ends = np.minimum(ends, n_offsets - 1)
        xs = boxes[:, 5:5 + n_offsets]
        offset_idx = np.arange(n_offsets)

        removed = np.zeros(n, dtype=bool)
        keep = []
        for i in range(n):
            if removed[i]:
                continue
            keep.append(order[i])
            if len(keep) == int(top_k):
                break
            rest = slice(i + 1, n)
            s = np.maximum(starts[i], starts[rest])
            e = np.minimum(ends[i], ends[rest])
            valid = (offset_idx[None, :] >= s[:, None]) & (offset_idx[None, :] <= e[:, None])
            diff = np.abs(xs[i][None, :] - xs[rest]) * valid
            counts = valid.sum(axis=1)
            mean_dist = diff.sum(axis=1) / np.maximum(counts, 1)
            suppress = (counts > 0) & (mean_dist < overlap) & (~removed[rest])
            removed[i + 1:][suppress] = True

        return np.asarray(keep, dtype=np.int64)

    def proposals_to_pred(self, proposals, img_w=640):
        """Post-NMS proposals -> [{'points': (N,2) normalized (x,y), 'conf': float}].

        Mirrors laneatt.py:proposals_to_pred, including its rule that a proposal not
        starting at the bottom of the image is extended upward only while x stays
        inside the frame. The original's `np.bool` (removed in numpy >= 1.24) is
        written as `bool` here.
        """
        if len(proposals) == 0:
            return []
        n_offsets = proposals.shape[1] - 5
        n_strips = n_offsets - 1
        anchor_ys = np.linspace(1.0, 0.0, n_offsets)

        lanes = []
        for lane in proposals:
            lane_xs = lane[5:].astype(np.float64) / img_w
            # Clamped: a degenerate proposal can produce start < 0, which would slice
            # from the end of the array instead of erroring. The torch original has
            # the same hazard; clamping only affects proposals that are already junk.
            start = int(round(float(lane[2]) * n_strips))
            start = max(0, min(start, n_offsets))
            length = int(round(float(lane[4])))
            end = max(min(start + length - 1, n_offsets - 1), -1)

            head = lane_xs[:start]
            inside = (head >= 0.0) & (head <= 1.0)
            # Reverse cumprod: keep the run of in-frame points adjacent to `start`.
            mask = ~(inside[::-1].cumprod()[::-1].astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2

            valid = lane_xs >= 0
            if valid.sum() <= 1:
                continue
            pts_x = lane_xs[valid][::-1]
            pts_y = anchor_ys[valid][::-1]
            lanes.append({
                "points": np.stack([pts_x, pts_y], axis=1),
                "conf": float(lane[1]),
                "start_x": float(lane[3]),
                "start_y": float(lane[2]),
            })
        return lanes
    
    def laneatt_decode(self, raw, conf_threshold=0.3, nms_thres=50.0, nms_topk=2, img_w=640):
        """Raw (1,1000,77) engine output -> list of lane dicts.

        Order matters: laneatt.py:nms scores with a softmax written to a *separate*
        tensor and slices the still-logit proposals, then decode() applies the softmax
        to the survivors. Softmaxing twice would squash every conf toward 0.5 and
        break the 0.5 acquire threshold downstream.
        """
        p = np.asarray(raw, dtype=np.float32).reshape(-1, np.shape(raw)[-1])
        scores = self._softmax2(p[:, :2])[:, 1]

        above = scores > conf_threshold
        p, scores = p[above], scores[above]
        if p.shape[0] == 0:
            return []

        p = p[self.lane_nms(p, scores, overlap=nms_thres, top_k=nms_topk)].copy()
        p[:, :2] = self._softmax2(p[:, :2])
        p[:, 4] = np.round(p[:, 4])
        return self.proposals_to_pred(p, img_w=img_w)

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
        return self.laneatt_decode(raw)

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
    
    def rg10_to_bgr(self, msg):
        """RG10 (V4L2 SRGGB10, 10-bit Bayer RGGB padded into 16-bit words) -> BGR8."""
        dtype = np.dtype(np.uint16).newbyteorder('>' if msg.is_bigendian else '<')
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)
        raw = raw[:, :msg.width]
        raw8 = (raw >> 2).astype(np.uint8)          # 10-bit (0-1023) -> 8-bit (0-255)
        return cv2.cvtColor(raw8, cv2.COLOR_BayerRG2BGR)

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
