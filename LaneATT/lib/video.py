import math
from lib.models.laneatt import LaneATT
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
    def __init__(self, model_wieghts, video_path = None, output_folder = None, view = True, frame_limit = 10000000000, device = torch.device("cuda:0"), conf_threshold = 0.5,  nms_thres = 50, nms_topk = 2):
        
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.model = LaneATT(backbone = "resnet18", topk_anchors = 1000, anchors_freq_path = "data/culane_anchors_freq.pt" )
        self.device = device
        self.to_tensor = ToTensor()
        self.view = view

        
        self.load_model(model_wieghts)
        self.logger = logging.getLogger("VideoInference")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.conf_threshold = conf_threshold
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk

    
    def load_model(self, wieghts):
        # `wieghts` is an already-built + weight-loaded nn.Module handed over by the
        # Runner (see Runner.get_model), so adopt it directly instead of loading a path.
        self.model = torch.load(wieghts)
        self.model = self.model.to(self.device)
        self.model.eval()
    
        
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
        output = self.model(frame, conf_threshold=self.conf_threshold, nms_thres=self.nms_thres, nms_topk=self.nms_topk)
        lanes = self.model.decode(output, as_lanes=True)[0]
        return lanes 
    
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
        """Classify detected lanes into the ego pair and compute the midpoints
        between them. Shared by show_frame (single image) and video_eval (per
        frame) so the classification logic only lives in one place.

        Returns (left_points, right_points, mid_points), or (None, None, None)
        if lanes weren't found on both sides of center this frame.
        """
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

        if not left_candidates or not right_candidates:
            return None, None, None

        # Closest lane to center on each side: largest x on the left, smallest x on the right.
        array_index_left = max(left_candidates, key=lambda t: t[0])[1]
        array_index_right = min(right_candidates, key=lambda t: t[0])[1]

        left_points = predictions[array_index_left]
        right_points = predictions[array_index_right]

        # Midpoints between the two ego lanes, one per shared y-row.
        left_by_y = {int(y): x for x, y in left_points}
        right_by_y = {int(y): x for x, y in right_points}
        shared_ys = sorted(set(left_by_y) & set(right_by_y))
        mid_points = np.array([[(left_by_y[y] + right_by_y[y]) / 2, y] for y in shared_ys], dtype=int)

        return left_points, right_points, mid_points

    def show_frame(self, image, predictions):
        left_points, right_points, mid_points = self.get_ego_lanes(image.shape[1], predictions)

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

    def image_eval(self):
        if self.video_path == None or Path(self.video_path).exists():
            self.logger.exception("Video Path is not defined")
        else: 
            print("No problem with Video Path")
        print(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        i = 0
        while i < 1:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            evaluation = self.frame_eval(frame)
            out = self.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])

            left, right, mid = self.show_frame(frame, out)
            

            # for pts in self.lanes_to_px(evaluation, frame.shape[1], frame.shape[0]):
            #     print(f"pts:{pts}")               
            #     for p0, p1 in zip(pts[:], pts[:]):
            #         print(f"p0: {p0.shape}, p0: {p0}")
            #         print(f"p1: {p1.shape}, {p1}")
                    
                    
            #         cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)
            
            
            i += 1
        
        
        
    def video_eval(self):
        if self.video_path == None or Path(self.video_path).exists():
            self.logger.exception("Video Path is not defined")
        else: 
            print("No problem with Video Path")
        print(f"Video Selected: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        out_stream = None
        if (self.output_folder != None):
            folder = Path(self.output_folder)
            folder_count = sum(1 for item in folder.iterdir() if item.is_dir())
            
            name = Path(self.video_path).stem

            new_folder_name = Path(f"video_{name}_{folder_count}")

            folder_path = folder / new_folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            video_name = Path("output.mp4")
            final_video_path = folder_path / video_name
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
                left_points, right_points, mid_points = self.get_ego_lanes(frame.shape[1], pts_all)
                if left_points is not None:
                    for p0, p1 in zip(left_points[:-1], left_points[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), (255, 0, 255), 4)   # magenta: left ego lane
                    for p0, p1 in zip(right_points[:-1], right_points[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), (255, 255, 0), 4)   # cyan: right ego lane
                    for x, y in mid_points:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)   # red: midpoint
            if (self.output_folder != None):
                out_stream.write(frame)
            
            if (i) % max(1, math.floor(local_frame_local / 10)) == 0:
                self.logger.info(f"Frame: {i}/{local_frame_local}, time: {str(time.time() - t1)}")
            
            i += 1
                
        t2 = time.time()
        self.logger.info("second: {}".format(t2-t1))

        
    

    