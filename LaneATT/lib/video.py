import math
import re
import torch
import cv2
import time as time
import logging
import numpy as np
import albumentations as A
from torchvision.transforms import ToTensor
from pathlib import Path
import logging
import matplotlib.pyplot as plt

class VideoInference():
    def __init__(self,model_archiecture, model_path, video_path = None, output_folder = None, view = True, frame_limit = 10000000000, device = torch.device("cuda:0"), conf_threshold = 0.5,  nms_thres = 50, nms_topk = 2, keep_threshold = 0.3, match_tolerance = 0.05):
        
        self.video_path = video_path
        
        
        if output_folder != None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.device = device
        self.to_tensor = ToTensor()
        self.view = view
        self.model_archiecture = model_archiecture

        
        self.load_model(model_path)
        # Per-y-row EMA of ego-lane width in pixels, learned while both edges are
        # visible. Width varies with y (perspective), so it must be per-row, not
        # a single scalar. Used to synthesize a missing edge from the visible one.
        self.lane_width_by_y = {}
        self.logger = logging.getLogger("VideoInference")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
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
    
    def set_video_path(self, video_path):
        self.video_path = video_path 
    def set_output_folder(self, output_path):
        self.output_folder = output_path
    def set_frame(self, frame):
        self.frame = frame
        
    def frame_eval(self,frame):
        frame = cv2.resize(frame, (640, 360))
        frame = self.to_tensor(frame)
        frame = frame.unsqueeze(0).to(self.device)
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
            # One edge, but no width prior learned yet this video.
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

    # def image_eval(self):
    #     if self.video_path is None or not Path(self.video_path).exists():
    #         raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
    #     print(self.video_path)
    #     cap = cv2.VideoCapture(self.video_path)
    #     i = 0
    #     while i < 1:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         evaluation = self.frame_eval(frame)
    #         out = self.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])

    #         left, right, mid = self.show_frame(frame, out)
            

    #         # for pts in self.lanes_to_px(evaluation, frame.shape[1], frame.shape[0]):
    #         #     print(f"pts:{pts}")               
    #         #     for p0, p1 in zip(pts[:], pts[:]):
    #         #         print(f"p0: {p0.shape}, p0: {p0}")
    #         #         print(f"p1: {p1.shape}, {p1}")
                    
                    
    #         #         cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)
            
            
    #         i += 1
        
        
        
    def video_eval(self):
        if self.video_path is None or not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
        print(f"Video Selected: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        out_stream = None
        if (self.output_folder != None):
            # <output_folder>/<video_stem>/run<K>/ where K = 1 + highest existing run
            # number for THIS video (a global folder count collided across videos).
            video_folder = Path(self.output_folder) / Path(self.video_path).stem
            video_folder.mkdir(parents=True, exist_ok=True)
            existing_runs = [int(m.group(1)) for d in video_folder.iterdir()
                             if d.is_dir() and (m := re.fullmatch(r'run(\d+)', d.name))]
            folder_path = video_folder / f"run{max(existing_runs, default=0) + 1}"
            folder_path.mkdir(parents=True, exist_ok=True)
            final_video_path = folder_path / "output.mp4"
            log_path = folder_path / "run.log"
            # Drop the FileHandler from a prior video_eval() call so each video's log
            # lines don't get duplicated into the next run's log.
            for h in list(self.logger.handlers):
                if isinstance(h, logging.FileHandler):
                    self.logger.removeHandler(h)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(fh)
            self.logger.info(f"video: {self.video_path}")
            self.logger.info(f"model checkpoint: {self.model_path}")
            self.logger.info(f"params: conf_threshold={self.conf_threshold}, "
                             f"nms_thres={self.nms_thres}, nms_topk={self.nms_topk}, "
                             f"frame_limit={self.frame_limit}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            print(f"Output Located: {final_video_path}")
            out_stream = cv2.VideoWriter(str(final_video_path), fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        i = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f'Video Frame rate: {str(fps)}')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"Total Frames: {total_frames}")
        duration_seconds = total_frames / fps if fps > 0 else 0
        self.logger.info(f"Video Duration: {duration_seconds}")
        
        if self.frame_limit > total_frames:
            local_frame_local = total_frames
        else:
            local_frame_local = self.frame_limit

        # Per-video state: width prior and hysteresis memory must not leak
        # between videos (different roads, different lane widths).
        self.lane_width_by_y = {}
        self.prev_lane_xs = []
        synth_frames = 0     # frames where one edge came from the width prior
        no_ego_frames = 0    # frames with no usable ego-lane pair

        t1 = time.time()
        
        
        
        
        
        while i < local_frame_local:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            evaluation = self.frame_eval(frame)
            if self.view:
                pts_all = self.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
                for pts in pts_all:
                    for p0, p1 in zip(pts[:-1], pts[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)

                # Colors are BGR (frame stays BGR end-to-end, see out_stream.write below).
                # A synthesized (width-prior) edge is drawn ORANGE instead of its
                # normal color so real vs inferred geometry is obvious in the video.
                left_points, right_points, mid_points, synthesized = self.get_ego_lanes(frame.shape[1], pts_all)

                if left_points is not None:
                    left_color = (0, 165, 255) if synthesized == 'left' else (255, 0, 255)    # orange / magenta
                    right_color = (0, 165, 255) if synthesized == 'right' else (255, 255, 0)  # orange / cyan
                    for p0, p1 in zip(left_points[:-1], left_points[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), left_color, 4)
                    for p0, p1 in zip(right_points[:-1], right_points[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), right_color, 4)
                    for x, y in mid_points:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)   # red: midpoint
                    synth_frames += synthesized is not None
                else:
                    no_ego_frames += 1
            if (self.output_folder != None):
                out_stream.write(frame)
            
            if (i) % max(1, math.floor(local_frame_local / 10)) == 0:
                self.logger.info(f"Frame: {i}/{local_frame_local}, time: {str(time.time() - t1)}")
            
            i += 1
                
        t2 = time.time()
        self.logger.info("second: {}".format(t2-t1))
        self.logger.info(f"ego-lane coverage: {i - no_ego_frames}/{i} frames "
                         f"({synth_frames} used a width-prior synthesized edge)")

        
    

    