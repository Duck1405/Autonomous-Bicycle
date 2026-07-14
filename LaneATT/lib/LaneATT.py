import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


class LaneATTInference():
    """LaneATT lane inference: model loading, per-frame eval with confidence
    hysteresis, and ego-lane extraction with the width-prior fallback.
    video.py owns the video loop and calls into this class."""

    def __init__(self, model_archiecture, model_path, device = torch.device("cuda:0"),
                 conf_threshold = 0.5, nms_thres = 50, nms_topk = 2,
                 keep_threshold = 0.3, match_tolerance = 0.05):
        self.device = device
        self.to_tensor = ToTensor()
        self.model_archiecture = model_archiecture
        self.load_model(model_path)
        # Per-y-row EMA of ego-lane width in pixels, learned while both edges are
        # visible. Width varies with y (perspective), so it must be per-row, not
        # a single scalar. Used to synthesize a missing edge from the visible one.
        self.lane_width_by_y = {}
        self.conf_threshold = conf_threshold
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk
        # Hysteresis: conf_threshold acquires a NEW lane; a lane matched (by
        # bottom-row x, within match_tolerance in normalized coords) to one
        # accepted in the previous frame survives down to keep_threshold.
        self.keep_threshold = keep_threshold
        self.match_tolerance = match_tolerance
        self.prev_lane_xs = []

    def load_model(self, wieghts):
        # `wieghts` is a checkpoint path; the bare architecture comes in separately
        # via model_archiecture and gets the checkpoint's state_dict loaded into it.
        self.model_path = wieghts
        state_dict = torch.load(
            wieghts,
            map_location='cpu'
        )['model']
        self.model_archiecture.load_state_dict(state_dict)
        self.model_archiecture.to(self.device)
        self.model_archiecture.eval()

    def update_paramaters(self, conf_threshold, nms_thres, nms_topk):
        self.conf_threshold = conf_threshold
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk

    def reset_video_state(self):
        # Per-video state: width prior and hysteresis memory must not leak
        # between videos (different roads, different lane widths).
        self.lane_width_by_y = {}
        self.prev_lane_xs = []

    def frame_eval(self, frame):
        frame = cv2.resize(frame, (640, 360))
        
        frame = self.to_tensor(frame)
        frame = frame.unsqueeze(0).to(self.device)
        print(f"frame output: {frame.shape}")
        # NMS runs at the LOW keep_threshold so borderline lanes reach the
        # hysteresis filter below instead of being discarded inside the model.
        output = self.model_archiecture(frame, conf_threshold=self.keep_threshold, nms_thres=self.nms_thres, nms_topk=self.nms_topk)
        lanes = self.model_archiecture.decode(output, as_lanes=True)[0]
        return self.filter_lanes_hysteresis(lanes)

    def filter_lanes_hysteresis(self, lanes):
        # Two-threshold acceptance: a new lane must score >= conf_threshold, but
        # one whose bottom-row x matches a lane accepted in the previous frame
        # survives down to keep_threshold. Stops real lanes from blinking in/out
        # when their score hovers around a single threshold; weak false positives
        # never cross the acquire bar, so they are never tracked.
        accepted, accepted_xs = [], []
        for lane in lanes:
            conf = float(lane.metadata['conf'])
            pts = lane.points   # normalized (x, y) in [0, 1]
            x_bottom = float(pts[np.argmax(pts[:, 1]), 0])
            tracked = any(abs(x_bottom - px) < self.match_tolerance for px in self.prev_lane_xs)
            if conf >= self.conf_threshold or (tracked and conf >= self.keep_threshold):
                accepted.append(lane)
                accepted_xs.append(x_bottom)
        self.prev_lane_xs = accepted_xs
        return accepted

    def lanes_to_px(self, lanes, w, h):
        out = []
        for lane in lanes:
            pts = lane.points.copy().astype(float)
            pts[:, 0] *= w          # Lane.points are normalized (x, y) in [0, 1]
            pts[:, 1] *= h
            out.append(pts.round().astype(int))
        return out

    def find_two_smallest(self, arr):
        # Initialize both variables to infinity
        smallest = second_smallest = float('inf')
        small_name = ""
        for num in arr.values():
            if num < smallest:
                second_smallest = smallest

                smallest = num
            elif num < second_smallest:
                # Change to 'elif num < second_smallest and num != smallest:'
                # if you strictly want the second smallest *distinct* element.
                second_smallest = num

        return [smallest, second_smallest]

    def get_ego_lanes(self, img_w, predictions):

        mid_point_x = img_w / 2

        # Classify each lane as a whole using ONE reference x per lane: its x at
        # the bottom-most row (closest to the car). Splitting a single lane's own
        # points into a left-sum/right-sum (the old approach) let a lane entirely
        # on one side win the *opposite* slot by default, since its unused side's
        # sum stayed at 0 -- always the minimum. Classifying by a single scalar
        # per lane avoids that trap.
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

    def show_frame(self, image, predictions):
        left_points, right_points, mid_points, synthesized = self.get_ego_lanes(image.shape[1], predictions)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if left_points is None:
            print("Could not find lanes on both sides of center; skipping midpoint.")
        else:
            print(f"left_points: {left_points}")
            print(f"right_points: {right_points}")
            plt.scatter(left_points[:, 0], left_points[:, 1], c='lime', s=10, label='left ego lane')
            plt.scatter(right_points[:, 0], right_points[:, 1], c='cyan', s=10, label='right ego lane')
            if len(mid_points) > 0:
                plt.scatter(mid_points[:, 0], mid_points[:, 1], c='red', s=10, label='midpoint')
            plt.legend(loc='upper right')

        return left_points, right_points, mid_points
