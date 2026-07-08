import math
import re
import torch
import cv2
import time as time
import logging
from pathlib import Path

from lib.LaneATT import LaneATTInference
from lib.yolo import YoloInference


class VideoInference():
    """Hub of the pipeline: owns the video loop, run folders, logging and drawing.
    Lane inference lives in LaneATT.py (LaneATTInference), object detection in
    yolo.py (YoloInference); parameters flow from here into those two."""

    def __init__(self, model_archiecture, model_path, video_path = None, output_folder = None, view = True, frame_limit = 10000000000, device = torch.device("cuda:0"), conf_threshold = 0.5,  nms_thres = 50, nms_topk = 2, keep_threshold = 0.3, match_tolerance = 0.05, yolo_path = None, yolo_conf = 0.5):

        self.video_path = video_path

        if output_folder != None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.view = view

        self.laneatt = LaneATTInference(model_archiecture, model_path, device=device,
                                        conf_threshold=conf_threshold, nms_thres=nms_thres,
                                        nms_topk=nms_topk, keep_threshold=keep_threshold,
                                        match_tolerance=match_tolerance)
        # yolo_path=None -> YoloInference falls back to its lib-relative default weights.
        self.yolo = YoloInference(yolo_path, conf_threshold=yolo_conf)

        self.logger = logging.getLogger("VideoInference")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

    def update_paramaters(self, conf_threshold, nms_thres, nms_topk):
        self.laneatt.update_paramaters(conf_threshold, nms_thres, nms_topk)

    def set_video_path(self, video_path):
        self.video_path = video_path
    def set_output_folder(self, output_path):
        self.output_folder = output_path
    def set_frame(self, frame):
        self.frame = frame
    def set_model(self, model_archiecture, model_path):
     
        self.laneatt.model_archiecture = model_archiecture
        self.laneatt.load_model(model_path)

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
            self.logger.info(f"model checkpoint: {self.laneatt.model_path}")
            self.logger.info(f"params: conf_threshold={self.laneatt.conf_threshold}, "
                             f"nms_thres={self.laneatt.nms_thres}, nms_topk={self.laneatt.nms_topk}, "
                             f"yolo_conf={self.yolo.conf_threshold}, frame_limit={self.frame_limit}")
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

        self.laneatt.reset_video_state()
        synth_frames = 0     # frames where one edge came from the width prior
        no_ego_frames = 0    # frames with no usable ego-lane pair
        lane_time = 0.0      # cumulative LaneATT inference seconds
        yolo_time = 0.0      # cumulative YOLO inference seconds

        t1 = time.time()

        while i < local_frame_local:
            ret, frame = cap.read()
            if not ret:
                break

            t = time.perf_counter()
            evaluation = self.laneatt.frame_eval(frame)
            lane_time += time.perf_counter() - t

            # YOLO sees the raw frame, before any lane drawing lands on it.
            t = time.perf_counter()
            yolo_results = self.yolo.infer(frame)
            yolo_time += time.perf_counter() - t

            if self.view:
                pts_all = self.laneatt.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
                for pts in pts_all:
                    for p0, p1 in zip(pts[:-1], pts[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)

                # Colors are BGR (frame stays BGR end-to-end, see out_stream.write below).
                # A synthesized (width-prior) edge is drawn ORANGE instead of its
                # normal color so real vs inferred geometry is obvious in the video.
                left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)

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

                frame = self.yolo.draw(frame, yolo_results)

            if (self.output_folder != None):
                out_stream.write(frame)

            if (i) % max(1, math.floor(local_frame_local / 10)) == 0:
                n = i + 1
                self.logger.info(f"Frame: {i}/{local_frame_local}, time: {str(time.time() - t1)}, "
                                 f"LaneATT: {lane_time:.1f}s ({1000 * lane_time / n:.0f} ms/frame), "
                                 f"YOLO: {yolo_time:.1f}s ({1000 * yolo_time / n:.0f} ms/frame)")

            i += 1

        t2 = time.time()
        self.logger.info("second: {}".format(t2-t1))
        if i > 0:
            self.logger.info(f"inference time over {i} frames: "
                             f"LaneATT {lane_time:.1f}s ({1000 * lane_time / i:.0f} ms/frame), "
                             f"YOLO {yolo_time:.1f}s ({1000 * yolo_time / i:.0f} ms/frame)")
        self.logger.info(f"ego-lane coverage: {i - no_ego_frames}/{i} frames "
                         f"({synth_frames} used a width-prior synthesized edge)")
