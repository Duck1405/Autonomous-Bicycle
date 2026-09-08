import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

from .trt_runner import CudaRT, TrtEngine
from .lane import Lane

class LaneATTNode(Node):
    def __init__(self):
        super().__init__('laneatt_node')
        self.engine = "/home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine"
        self.warmup = 50
        self.CudaRT = CudaRT()
        self.trt_engine = TrtEngine(self.engine, self.CudaRT)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/stereo/left/image_raw', self.image_callback, 1)  # Ros2: "/left/raw", "/right/raw"
        # raw_camera left: /dev/video0, right: /dev/video1
        self.left_pub = self.create_publisher(Float32MultiArray, '/laneatt/left_lane', 1)
        self.right_pub = self.create_publisher(Float32MultiArray, '/laneatt/right_lane', 1)
        self.lane_width_by_y = {}

        for _ in range(self.warmup):
            self.frame_eval(np.zeros((360, 640, 3), dtype=np.uint8))

        self.get_logger().info('LaneATT node started')
    
    
    def _softmax2(self, logits):
        """Row-wise softmax over a (N, 2) block, max-subtracted for stability."""
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def nms(self, proposals, scores, overlap=50.0, top_k=4):
        """TensorRT-side counterpart of the notebook model's nms method.

        Returns kept indices into `proposals`, best score first.

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
        """Post-NMS proposals -> Lane objects with normalized (N,2) points.

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
            lanes.append(Lane(
                points=np.stack([pts_x, pts_y], axis=1),
                metadata={
                    "conf": float(lane[1]),
                    "start_x": float(lane[3]),
                    "start_y": float(lane[2]),
                },
            ))
        return lanes
    
    def decode(self, raw, conf_threshold=0.3, nms_thres=50.0, nms_topk=2, img_w=640):
        """Raw (1,1000,77) engine output -> list of Lane objects.

        Named after the notebook model's decode; this TensorRT version also
        filters confidence and runs NMS, which PyTorch runs in its forward pass.

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

        p = p[self.nms(p, scores, overlap=nms_thres, top_k=nms_topk)].copy()
        p[:, :2] = self._softmax2(p[:, :2])
        p[:, 4] = np.round(p[:, 4])
        return self.proposals_to_pred(p, img_w=img_w)

    def frame_eval(self, frame):
        """Preprocess one BGR frame, run TensorRT, and return Lane objects.

        Matches the notebook's frame_eval entry point: resize to 640x360,
        scale to [0, 1], and arrange as (1, 3, 360, 640), keeping BGR order.
        decode applies confidence filtering and NMS to the raw engine output.
        """
        frame = cv2.resize(frame, (640, 360))
        frame = frame.astype(np.float32) / 255.0
        frame = np.ascontiguousarray(frame.transpose(2, 0, 1)[None])
        outputs = self.trt_engine.infer(frame)
        raw = next(iter(outputs.values()))
        return self.decode(raw)

    @staticmethod
    def _bottom_x(lane):
        """Normalized x where a lane's polyline is closest to the camera (largest y)."""
        pts = lane.points
        return float(pts[np.argmax(pts[:, 1]), 0])

    def split_left_right(self, lanes):
        """Lane objects -> (left, right), ordered by bottom-row x position.

        decode keeps at most 2 lanes by default (nms_topk=2), which for
        ego-lane detection are the left and right boundary. Missing side(s) come
        back as None rather than guessing.
        """
        if not lanes:
            return None, None
        scored = sorted(lanes, key=self._bottom_x)
        if len(scored) == 1:
            return (scored[0], None) if self._bottom_x(scored[0]) < 0.5 else (None, scored[0])
        return scored[0], scored[-1]

    
    
    def rg10_to_bgr(self, msg):
        """RG10 (V4L2 SRGGB10, 10-bit Bayer RGGB padded into 16-bit words) -> BGR8."""
        dtype = np.dtype(np.uint16).newbyteorder('>' if msg.is_bigendian else '<')
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)
        raw = raw[:, :msg.width]
        raw8 = (raw >> 2).astype(np.uint8)          # 10-bit (0-1023) -> 8-bit (0-255)
        return cv2.cvtColor(raw8, cv2.COLOR_BayerRG2BGR)
    
    def get_ego_lanes(self, img_w, predictions):

        mid_point_x = img_w / 2


        left_candidates = []   # (x_bottom, lane_index), x_bottom < mid
        right_candidates = []  # (x_bottom, lane_index), x_bottom >= mid
        for i, lane in enumerate(predictions):
            bottom_idx = np.argmax(lane[:, 1])  # largest y = nearest the car
            x_bottom = lane[bottom_idx, 0]
            if x_bottom < mid_point_x:
                left_candidates.append((x_bottom, i))
            else:
                right_candidates.append((x_bottom, i))

        if not left_candidates and not right_candidates:
            return None, None, None, None

        # Closest lane to center on each side: largest x on the left, smallest x on the right.
        left_points = right_points = None
        if left_candidates:
            left_points = predictions[max(left_candidates, key=lambda t: t[0])[1]]
        if right_candidates:
            right_points = predictions[min(right_candidates, key=lambda t: t[0])[1]]

        # `synthesized` names the edge ('left'/'right') that was inferred from the
        # width prior instead of detected, or None when both edges are real.
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
                return None, None, None, None   # too little prior overlap to trust
            synth = np.array(synth).round().astype(int)
            if left_points is not None:
                right_points, synthesized = synth, 'right'
            else:
                left_points, synthesized = synth, 'left'
        else:
            return None, None, None, None

        # Midpoints between the two ego lanes, one per shared y-row.
        left_by_y = {int(y): x for x, y in left_points}
        right_by_y = {int(y): x for x, y in right_points}
        shared_ys = sorted(set(left_by_y) & set(right_by_y))
        mid_points = np.array([[(left_by_y[y] + right_by_y[y]) / 2, y] for y in shared_ys], dtype=int)

        return left_points, right_points, mid_points, synthesized

    @staticmethod
    def lanes_to_px(lanes, w, h):
        """Convert normalized Lane points to rounded image pixel coordinates."""
        return [(lane.points * np.array([w, h])).round().astype(int)
                for lane in lanes]

    @staticmethod
    def points_to_lane(points, img_w, img_h, synthesized=False):
        """Convert an ego boundary in pixels to a normalized Lane, or None.

        Pixel rounding can repeat y rows. Keep the first point per row and
        sort top-to-bottom so Lane's spline has strictly increasing y values.
        No model confidence is assigned to reconstructed geometry.
        """
        if points is None:
            return None
        if img_w <= 0 or img_h <= 0:
            raise ValueError("Image width and height must be positive")
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return None
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected pixel points with shape (N, 2)")
        if not np.isfinite(points).all():
            raise ValueError("Lane points must be finite")
        _, indices = np.unique(points[:, 1], return_index=True)
        points = points[indices]
        if len(points) < 2:
            return None
        return Lane(points=points / np.array([img_w, img_h]),
                    metadata={"synthesized": bool(synthesized)})

    @staticmethod
    def _to_msg(lane):
        msg = Float32MultiArray()
        if lane is not None:
            msg.data = lane.points.astype(np.float32).flatten().tolist()
        return msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        width = frame.shape[1]
        height = frame.shape[0]
        
        evaluation = self.frame_eval(frame)
        
        pts_all = self.lanes_to_px(evaluation, width, height)
        left_points, right_points, mid_points, synthesized = self.get_ego_lanes(width, pts_all)
        left = self.points_to_lane(left_points, width, height,
                                   synthesized=synthesized == 'left')
        right = self.points_to_lane(right_points, width, height,
                                    synthesized=synthesized == 'right')
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
