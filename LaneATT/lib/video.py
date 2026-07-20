import math
import re
import torch
import cv2
import time as time
import logging
from pathlib import Path

from lib.LaneATT import LaneATTInference
from lib.lanenet_infer import LaneNetInference
from lib.yolo import YoloInference
from lib.depth import DepthInference
from lib.angle import Angle
from PIL import Image
import numpy as np

class VideoInference():
    """Hub of the pipeline: owns the video loop, run folders, logging and drawing.
    model_type picks the lane model: "laneATT" -> LaneATT.py (LaneATTInference),
    "laneNet" -> lanenet_infer.py (LaneNetInference); object detection in
    yolo.py (YoloInference); parameters flow from here into those."""

    def __init__(self, model_type = "laneATT", model_archiecture = None, model_path = None, hnet_path = None, video_path = None, output_folder = None, view = True, frame_limit = 10000000000, device = torch.device("cuda:0"), conf_threshold = 0.5,  nms_thres = 50, nms_topk = 2, keep_threshold = 0.3, match_tolerance = 0.05, yolo_path = None, yolo_conf = 0.5, yolo_iou = 0.15):

        self.video_path = video_path

        if output_folder != None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.view = view

        self.device = device

        # model_type picks the lane model: "laneNet" uses model_path (+ hnet_path),
        # "laneATT" uses model_archiecture + model_path.
        self.model_type = model_type
        self.laneatt = None
        self.lanenet = None
        if model_type == "laneNet":
            self.lanenet = LaneNetInference(model_path, hnet_path=hnet_path, device=device)
        else:
            self.laneatt = LaneATTInference(model_archiecture, model_path, device=device,
                                            conf_threshold=conf_threshold, nms_thres=nms_thres,
                                            nms_topk=nms_topk, keep_threshold=keep_threshold,
                                            match_tolerance=match_tolerance)
        # yolo_path=None -> YoloInference falls back to its lib-relative default weights.
        self.yolo = YoloInference(yolo_path, conf_threshold=yolo_conf, iou_threshold=yolo_iou,
                                  device=device)
        # Monocular relative depth (Depth-Anything-V2); picks cuda/cpu itself.
        self.depth = DepthInference()
        # Steering + lead-vehicle layer over get_ego_lanes()/YOLO (laneATT path only;
        # the laneNet branch doesn't produce get_ego_lanes()-shaped midpoints).
        self.angle = Angle(vehicle_class_id=1)

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
        # Swap in a LaneATT checkpoint and make LaneATT the active lane model.
        if self.laneatt is None:
            self.laneatt = LaneATTInference(model_archiecture, model_path, device=self.device)
        else:
            self.laneatt.model_archiecture = model_archiecture
            self.laneatt.load_model(model_path)
        self.model_type = "laneATT"

    def set_lanenet(self, model_path, hnet_path=None):
        # Load LaneNet (+ optional H-Net) and make it the active lane model.
        self.lanenet = LaneNetInference(model_path, hnet_path=hnet_path, device=self.device)
        self.model_type = "laneNet"

    def speed_eval(self, speed):
        # First Base Speed on if edges are found 
    
        return

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
            if self.model_type == "laneNet":
                self.logger.info(f"model checkpoint: {self.lanenet.model_path} "
                                 f"(hnet: {self.lanenet.hnet_path})")
                self.logger.info(f"params: yolo_conf={self.yolo.conf_threshold}, "
                                 f"yolo_iou={self.yolo.iou_threshold}, "
                                 f"frame_limit={self.frame_limit}")
            else:
                self.logger.info(f"model checkpoint: {self.laneatt.model_path}")
                self.logger.info(f"params: conf_threshold={self.laneatt.conf_threshold}, "
                                 f"nms_thres={self.laneatt.nms_thres}, nms_topk={self.laneatt.nms_topk}, "
                                 f"keep_threshold={self.laneatt.keep_threshold}, "
                                 f"match_tolerance={self.laneatt.match_tolerance}, "
                                 f"yolo_conf={self.yolo.conf_threshold}, yolo_iou={self.yolo.iou_threshold}, "
                                 f"frame_limit={self.frame_limit}")
                self.logger.info(f"angle params: hfov={self.angle.assumed_hfov_deg}, "
                                 f"stanley_gain={self.angle.stanley_gain}, "
                                 f"nominal_speed={self.angle.nominal_speed}, "
                                 f"smoothing_alpha={self.angle.smoothing_alpha}, "
                                 f"hold_decay={self.angle.hold_decay}, "
                                 f"max_extrapolation_px={self.angle.max_extrapolation_px}, "
                                 f"vehicle_class_id={self.angle.vehicle_class_id}, "
                                 f"lane_width_m={self.angle.lane_width_m}")
            self.logger.info(f"depth model: {self.depth.checkpoint} (device: {self.depth.device})")
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

        use_lanenet = self.model_type == "laneNet"
        lane_label = "LaneNet" if use_lanenet else "LaneATT"
        self.logger.info(f"devices: {lane_label} {self.device}, "
                         f"YOLO {self.yolo.model.device}, Depth {self.depth.device}")
        if not use_lanenet:
            self.laneatt.reset_video_state()
            self.angle.reset_video_state()
        synth_frames = 0     # frames where one edge came from the width prior
        no_ego_frames = 0    # frames with no usable ego-lane pair
        no_steer_frames = 0  # frames with no steering (even the held value expired)
        lead_frames = 0      # frames with a lead vehicle (CIPV) selected
        lane_time = 0.0      # cumulative lane-model inference seconds
        yolo_time = 0.0      # cumulative YOLO inference seconds
        depth_time = 0.0     # cumulative depth-model inference seconds

        t1 = time.time()

        while i < local_frame_local:
            ret, frame = cap.read()
            if not ret:
                break

            t = time.perf_counter()
            if use_lanenet:
                evaluation = self.lanenet.frame_eval(frame)
            else:
                evaluation = self.laneatt.frame_eval(frame)
            lane_time += time.perf_counter() - t

            # YOLO sees the raw frame, before any lane drawing lands on it.
            t = time.perf_counter()
            yolo_results = self.yolo.infer(frame)
            yolo_time += time.perf_counter() - t

            # Depth also runs on the raw frame (relative depth, unused downstream yet).
            # t = time.perf_counter()
            # depth_results = self.depth.infer(frame)
            # depth_time += time.perf_counter() - t

            if self.view:
                # if use_lanenet:
                #     frame = self.lanenet.draw(frame, evaluation)
                # else:
                #     pts_all = self.laneatt.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
                #     for pts in pts_all:
                #         for p0, p1 in zip(pts[:-1], pts[1:]):
                #             cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)

                #     # Colors are BGR (frame stays BGR end-to-end, see out_stream.write below).
                #     # A synthesized (width-prior) edge is drawn ORANGE instead of its
                #     # normal color so real vs inferred geometry is obvious in the video.
                #     left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)

                #     if left_points is not None:
                #         left_color = (0, 165, 255) if synthesized == 'left' else (255, 0, 255)    # orange / magenta
                #         right_color = (0, 165, 255) if synthesized == 'right' else (255, 255, 0)  # orange / cyan
                #         for p0, p1 in zip(left_points[:-1], left_points[1:]):
                #             cv2.line(frame, tuple(p0), tuple(p1), left_color, 4)
                #         for p0, p1 in zip(right_points[:-1], right_points[1:]):
                #             cv2.line(frame, tuple(p0), tuple(p1), right_color, 4)
                #         for x, y in mid_points:
                #             cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)   # red: midpoint
                #         synth_frames += synthesized is not None
                #     else:
                #         no_ego_frames += 1
                pts_all = self.laneatt.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
                for pts in pts_all:
                    for p0, p1 in zip(pts[:-1], pts[1:]):
                        cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)

                # Colors are BGR (frame stays BGR end-to-end, see out_stream.write below).
                # A synthesized (width-prior) edge is drawn ORANGE instead of its
                # normal color so real vs inferred geometry is obvious in the video.
                left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)

                if left_points is not None and right_points is not None and mid_points is not None:
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

                steering = self.angle.compute_steering(
                    mid_points, frame.shape[1], frame.shape[0],
                    left_points=left_points, right_points=right_points)
                ego_vehicle = self.angle.select_ego_vehicle(yolo_results, left_points, right_points)
                no_steer_frames += steering is None
                lead_frames += ego_vehicle is not None

                frame = self.yolo.draw(frame, yolo_results)
                frame = self.angle.draw_overlay(frame, steering, ego_vehicle)

            if (self.output_folder != None):
                out_stream.write(frame)

            # Live console progress: current frame / frames this run (and the
            # video's true total when frame_limit cut it short).
            progress = f"Frame {i + 1}/{local_frame_local}"
            if local_frame_local != total_frames:
                progress += f" (video has {total_frames})"
            # Whole-pipeline throughput so far (all models + drawing + writing).
            progress += f", {(i + 1) / (time.time() - t1):.2f} FPS"
            print(f"\r{progress}", end="", flush=True)

            if (i) % max(1, math.floor(local_frame_local / 10)) == 0:
                n = i + 1
                self.logger.info(f"Frame: {i}/{local_frame_local}, time: {str(time.time() - t1)}, "
                                 f"{lane_label}: {lane_time:.1f}s ({1000 * lane_time / n:.0f} ms/frame), "
                                 f"YOLO: {yolo_time:.1f}s ({1000 * yolo_time / n:.0f} ms/frame), "
                                 f"Depth: {depth_time:.1f}s ({1000 * depth_time / n:.0f} ms/frame)")

            i += 1

        print()   # end the \r progress line
        t2 = time.time()
        self.logger.info("second: {}".format(t2-t1))
        if i > 0 and t2 > t1:
            self.logger.info(f"pipeline throughput: {i / (t2 - t1):.2f} FPS "
                             f"({1000 * (t2 - t1) / i:.0f} ms/frame wall)")
        if i > 0:
            self.logger.info(f"inference time over {i} frames: "
                             f"{lane_label} {lane_time:.1f}s ({1000 * lane_time / i:.0f} ms/frame), "
                             f"YOLO {yolo_time:.1f}s ({1000 * yolo_time / i:.0f} ms/frame), "
                             f"Depth {depth_time:.1f}s ({1000 * depth_time / i:.0f} ms/frame)")
        if not use_lanenet:
            self.logger.info(f"ego-lane coverage: {i - no_ego_frames}/{i} frames "
                             f"({synth_frames} used a width-prior synthesized edge)")
            self.logger.info(f"steering coverage: {i - no_steer_frames}/{i} frames, "
                             f"lead vehicle selected on {lead_frames}/{i} frames")

    def image_eval(self, frame_number):
        """Evaluate ONE frame of self.video_path (by index, 0-based) through the
        full pipeline and save <output_folder>/<video_stem>/frame_<N>/<N>.jpg
        plus a run.log with per-stage timings. LaneATT path only."""
        if self.model_type == "laneNet":
            raise ValueError("image_eval only supports the laneATT model path")
        if self.video_path is None or not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
        if self.output_folder is None:
            raise ValueError("image_eval needs output_folder set to save results")

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (0 <= frame_number < total_frames):
            cap.release()
            raise ValueError(f"frame {frame_number} out of range (video has {total_frames} frames)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        convert = np.array(frame)
        print(f"Get Image numpy: {convert}")
        print(f"Get Image Shape: {convert.shape}")
        
        if not ret:
            raise RuntimeError(f"Could not decode frame {frame_number} of {self.video_path}")

        folder_path = Path(self.output_folder) / Path(self.video_path).stem / f"frame_{frame_number}"
        folder_path.mkdir(parents=True, exist_ok=True)
        # Same handler-swap as video_eval so log lines don't duplicate across calls.
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)
        fh = logging.FileHandler(folder_path / "run.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)
        self.logger.info(f"video: {self.video_path}, frame: {frame_number}/{total_frames}")
        self.logger.info(f"devices: LaneATT {self.device}, "
                         f"YOLO {self.yolo.model.device}, Depth {self.depth.device}")
        self.logger.info(f"model checkpoint: {self.laneatt.model_path}")
        self.logger.info(f"params: conf_threshold={self.laneatt.conf_threshold}, "
                         f"nms_thres={self.laneatt.nms_thres}, nms_topk={self.laneatt.nms_topk}, "
                         f"keep_threshold={self.laneatt.keep_threshold}, "
                         f"match_tolerance={self.laneatt.match_tolerance}, "
                         f"yolo_conf={self.yolo.conf_threshold}, yolo_iou={self.yolo.iou_threshold}")

        # A single frame has no history: clear the width-prior/hysteresis and the
        # steering EMA so the result is pure per-frame, not leftovers from a video.
        self.laneatt.reset_video_state()
        self.angle.reset_video_state()

        t0 = time.perf_counter()
        evaluation = self.laneatt.frame_eval(frame)
        lane_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        yolo_results = self.yolo.infer(frame)
        yolo_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        depth_results = self.depth.infer(frame)
        depth_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts_all = self.laneatt.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
        for pts in pts_all:
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)
        left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)
        if left_points is not None and right_points is not None and mid_points is not None:
            left_color = (0, 165, 255) if synthesized == 'left' else (255, 0, 255)    # orange / magenta
            right_color = (0, 165, 255) if synthesized == 'right' else (255, 255, 0)  # orange / cyan
            for p0, p1 in zip(left_points[:-1], left_points[1:]):
                cv2.line(frame, tuple(p0), tuple(p1), left_color, 4)
            for p0, p1 in zip(right_points[:-1], right_points[1:]):
                cv2.line(frame, tuple(p0), tuple(p1), right_color, 4)
            for x, y in mid_points:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)   # red: midpoint

        steering = self.angle.compute_steering(
            mid_points, frame.shape[1], frame.shape[0],
            left_points=left_points, right_points=right_points)
        ego_vehicle = self.angle.select_ego_vehicle(yolo_results, left_points, right_points)
        frame = self.yolo.draw(frame, yolo_results)
        frame = self.angle.draw_overlay(frame, steering, ego_vehicle)
        angle_time = time.perf_counter() - t0

        self.logger.info(f"steering: {steering}")
        self.logger.info(f"ego_vehicle: {ego_vehicle}")
        self.logger.info(f"timing: LaneATT {1000 * lane_time:.1f} ms, "
                         f"YOLO {1000 * yolo_time:.1f} ms, "
                         f"Depth {1000 * depth_time:.1f} ms, "
                         f"angle+draw {1000 * angle_time:.1f} ms, "
                         f"total {1000 * (lane_time + yolo_time + depth_time + angle_time):.1f} ms")

        output_file = folder_path / f"{frame_number}.jpg"
        cv2.imwrite(str(output_file), frame)
        self.logger.info(f"saved: {output_file}")
        print(f"Output Located: {output_file}")
        return steering, ego_vehicle
