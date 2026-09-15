import math
import re
from contextlib import nullcontext
import cv2
import time as time
import logging
from pathlib import Path

from .ego_lanes import get_ego_lanes2
from .angle import Angle
import numpy as np
import sys


class DecodedYoloDrawing:
    """Draw decoded ONNX/TensorRT detections without importing Ultralytics."""

    names = ("person", "vehicle", "traffic-light", "stop-sign")
    colors = ((0, 255, 0), (255, 160, 0), (0, 215, 255), (0, 0, 255))

    def draw(self, frame, results):
        for p1, p2, conf, cls in results:
            color = self.colors[cls % len(self.colors)]
            name = self.names[cls] if 0 <= cls < len(self.names) else str(cls)
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (p1[0], max(p1[1] - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame


class VideoInference():
    """Hub of the pipeline: owns the video loop, run folders, logging and drawing.
    model_type picks the lane model: "laneATT" -> LaneATT.py (LaneATTInference),
    "laneNet" -> lanenet_infer.py (LaneNetInference); object detection in
    yolo.py (YoloInference); parameters flow from here into those."""

    def __init__(self, model_type = "laneATT", model_archiecture = None, model_path = None, hnet_path = None, video_path = None, output_folder = None, view = True, 
                 frame_limit = 10000000000, device = "cuda:0", conf_threshold = 0.5,  nms_thres = 50, nms_topk = 4,
                 keep_threshold = 0.3, match_tolerance = 0.05, yolo_path = None, yolo_conf = 0.5, yolo_iou = 0.15, initialize_models=True):


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
        
        self.nms_topk = nms_topk
        self.nms_thres = nms_thres
        self.keep_threshold = keep_threshold
        self.match_tolerance = match_tolerance
        self.yolo_iou = yolo_iou
        self.model_archiecture = model_archiecture
        self.model_path = model_path
        self.device = device 
        self.conf_threshold = conf_threshold
        
        self.yolo_path = yolo_path
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        
        
        print(f"conf_threshold: {conf_threshold}")
        print(f"nms_thres: {nms_thres}")
        print(f"nmms topK: {nms_topk}")
        
        
        # if model_type == "laneNet":
        #     self.lanenet = LaneNetInference(model_path, hnet_path=hnet_path, device=device)

        self.laneatt = None
        # yolo_path=None -> YoloInference falls back to its lib-relative default weights.
        self.yolo = None
        
        self.inference_context = nullcontext
        if initialize_models:
            import torch
            self.inference_context = torch.no_grad
            self.update_laneATT()
            self.update_yolo()
        # Monocular relative depth (Depth-Anything-V2); picks cuda/cpu itself.
        # self.depth = DepthInference()
        # Steering + lead-vehicle layer over get_ego_lanes()/YOLO (laneATT path only;
        # the laneNet branch doesn't produce get_ego_lanes()-shaped midpoints).
        self.angle = Angle(vehicle_class_id=1)

        self.logger = logging.getLogger("VideoInference")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        
    def update_laneATT(self):
        from .LaneATT import LaneATTInference
        self.laneatt = LaneATTInference(self.model_archiecture,  self.model_path, device=self.device,
                                            conf_threshold=self.conf_threshold, nms_thres=self.nms_thres,
                                            nms_topk=self.nms_topk, keep_threshold=self.keep_threshold,
                                            match_tolerance=self.match_tolerance, )
    def update_yolo(self):
        from .yolo import YoloInference
        self.yolo = YoloInference(self.yolo_path, conf_threshold=self.yolo_conf, iou_threshold=self.yolo_iou,device=self.device)
        
    
    def update_nms_thres(self, nms_thres):
        self.nms_thres = nms_thres
        
        
    def update_nms_topk(self, nms_topk):
        self.nms_topk = nms_topk
    def update_keep_threshold(self, keep_threshold):
        self.keep_threshold = keep_threshold
    def update_match_tolerance(self, match_tolerance):
         self.match_tolerance = match_tolerance
    def update_yolo_iou(self, yolo_iou):
        self.yolo_iou = yolo_iou
        
    # def update_paramaters(self, conf_threshold, nms_thres, nms_topk):
    #     self.laneatt.update_paramaters(conf_threshold, nms_thres, nms_topk)

    def set_video_path(self, video_path):
        self.video_path = video_path
    def set_output_folder(self, output_path):
        self.output_folder = output_path
    def set_frame(self, frame):
        self.frame = frame
    def set_model(self, model_archiecture, model_path):
        from .LaneATT import LaneATTInference
        # Swap in a LaneATT checkpoint and make LaneATT the active lane model.
        if self.laneatt is None:
            self.laneatt = LaneATTInference(model_archiecture, model_path, device=self.device)
        else:
            self.laneatt.model_archiecture = model_archiecture
            self.laneatt.load_model(model_path)
        self.model_type = "laneATT"

    def set_lanenet(self, model_path, hnet_path=None):
        from .lanenet_infer import LaneNetInference
        # Load LaneNet (+ optional H-Net) and makepath it the active lane model.
        self.lanenet = LaneNetInference(model_path, hnet_path=hnet_path, device=self.device)
        self.model_type = "laneNet"

    def speed_eval(self, speed):
        # First Base Speed on if edges are found 
    
        return
    
    def get_frame(self, frame, split="new", evaluation=None, yolo_results=None):

        t0 = time.perf_counter()
        if evaluation is None:
            with self.inference_context():
                evaluation = self.laneatt.frame_eval(frame)
        lane_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        if yolo_results is None and self.yolo is not None:
            yolo_results = self.yolo.infer(frame)
        yolo_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        # depth_results = self.depth.infer(frame)
        # depth_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts_all = [(lane.points * np.array([frame.shape[1], frame.shape[0]])).round().astype(int)
                   for lane in evaluation]
        # for pts in pts_all:
        #     for p0, p1 in zip(pts[:-1], pts[1:]):
        #         cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)
                
  
        if split == "base":
            left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)
        elif split == "new":
            left_points, right_points, mid_points, synthesized = get_ego_lanes2(frame.shape[1], pts_all)
        else:
            raise ValueError(f"Unknown split: {split!r}; expected base or new")
        # print(f"left_points: {left_points}")
        # print(f"right_points: {right_points}")
        # print(f"mid_points: {mid_points}")
        
        if left_points is not None and right_points is not None and mid_points is not None:
            left_color = (255, 0,0) 
            right_color = (0,0,255)
            (0, 165, 255) if synthesized == 'right' else (255, 255, 0)  # orange / cyan
            for p0, p1 in zip(left_points[:-1], left_points[1:]):
                cv2.line(frame, tuple(p0), tuple(p1), left_color, 4)
            for p0, p1 in zip(right_points[:-1], right_points[1:]):
                cv2.line(frame, tuple(p0), tuple(p1), right_color, 4)
            for x, y in mid_points:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)   # red: midpoint
                
        if self.yolo is not None:
            frame = self.yolo.draw(frame, yolo_results)

        return frame

    def video_eval(self, *, start_frame=0, output_path=None, codec="mp4v",
                   frame_processor=None, warmup=None, render=True):
        """Read, infer and save a video using either backend.

        frame_processor(frame) returns an annotated frame (or None for timing
        without rendering). The default runs the PyTorch backends. warmup(first)
        runs before timing and does not consume any output frames.
        """
        if self.frame_limit <= 0 or start_frame < 0:
            raise ValueError("frame_limit must be positive and start_frame non-negative")
        if self.video_path is None or not Path(self.video_path).is_file():
            raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        writer = handler = None
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            # Decode skipped frames instead of trusting AVI seeking/index metadata.
            for _ in range(start_frame):
                if not cap.grab():
                    raise RuntimeError(f"Video ends before start frame {start_frame}")
            ok, first = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {start_frame}: {self.video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            if output_path is None and render:
                if self.output_folder is None:
                    raise ValueError("video_eval requires output_folder or output_path")
                folder = Path(self.output_folder) / Path(self.video_path).stem
                folder.mkdir(parents=True, exist_ok=True)
                runs = [int(m.group(1)) for d in folder.iterdir()
                        if d.is_dir() and (m := re.fullmatch(r"run(\d+)", d.name))]
                folder = folder / f"run{max(runs, default=0) + 1}"
                output_path = folder / "output.mp4"
            if render:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                (output_path.parent / "frames").mkdir(exist_ok=True)
                handler = logging.FileHandler(output_path.parent / "run.log")
                handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                self.logger.addHandler(handler)
                fourcc = getattr(cv2.VideoWriter, "fourcc", None)
                if fourcc is None:
                    fourcc = cv2.VideoWriter_fourcc
                h, w = first.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc(*codec), fps, (w * 2, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {output_path} ({codec})")
                print(f"Output: {output_path} ({w * 2}x{h}, {fps:.2f} FPS)")
            if self.laneatt is not None:
                self.laneatt.reset_video_state()
            self.angle.reset_video_state()
            if warmup is not None:
                warmup(first)
            self.logger.info("video=%s start_frame=%s frame_limit=%s model=%s device=%s",
                             self.video_path, start_frame, self.frame_limit, self.model_path, self.device)
            for name, backend in (("LaneATT", self.laneatt), ("YOLO", self.yolo)):
                if backend is not None:
                    self.logger.info("%s params=%s", name, {k: getattr(backend, k, None) for k in
                                     ("conf_threshold", "nms_thres", "nms_topk", "keep_threshold",
                                      "match_tolerance", "iou_threshold")})
            done = 0
            read_seconds = write_seconds = 0.0
            started = time.perf_counter()
            while done < self.frame_limit:
                t0 = time.perf_counter()
                if done == 0:
                    frame = first
                else:
                    ok, frame = cap.read()
                    if not ok:
                        print(f"End of video after {done} frames")
                        break
                read_seconds += time.perf_counter() - t0
                original = frame.copy() if render else None
                if frame_processor is None:
                    annotated = self.get_frame(frame)
                else:
                    annotated = frame_processor(frame)
                if writer is not None:
                    t0 = time.perf_counter()
                    writer.write(cv2.hconcat([original, annotated]))
                    write_seconds += time.perf_counter() - t0
                done += 1
                if done % 10 == 0 or done == self.frame_limit:
                    print(f"\r{done}/{self.frame_limit} frames", end="", flush=True)
            elapsed = time.perf_counter() - started
            stats = dict(frames=done, start_frame=start_frame, read_seconds=read_seconds,
                         write_seconds=write_seconds, elapsed_seconds=elapsed,
                         output=str(output_path) if render else None)
            self.logger.info("Completed: %s", stats)
            print(f"\nCompleted {done} frames in {elapsed:.2f}s")
            return stats
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if handler is not None:
                self.logger.removeHandler(handler)
                handler.close()

    def image_eval(self, frame_number):

        if self.video_path is None or not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
        print(f"Video Selected: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        image_folder = Path(self.output_folder) / Path(self.video_path).stem / f"frame_{frame_number}"
        image_folder.mkdir(parents=True, exist_ok=True)
        existing_runs = [int(m.group(1)) for d in image_folder.iterdir()
                            if d.is_dir() and (m := re.fullmatch(r'run(\d+)', d.name))]
        
        folder_path = image_folder / f"run{max(existing_runs, default=0) + 1}"
        folder_path.mkdir(parents=True, exist_ok=True)
        final_video_path = folder_path / "output.jpg"
        log_path = folder_path / "run.log"
        
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(image_folder / "run.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))
    
       
        self.logger.addHandler(fh)
        self.logger.info(f"video: {self.video_path}, frame: {frame_number}/{total_frames}")
        self.logger.info(f"devices: LaneATT {self.device}, "
                         f"YOLO {self.yolo.model.device}")
        self.logger.info(f"model checkpoint: {self.laneatt.model_path}")
        self.logger.info(f"params: conf_threshold={self.laneatt.conf_threshold}, "
                         f"nms_thres={self.laneatt.nms_thres}, nms_topk={self.laneatt.nms_topk}, "
                         f"keep_threshold={self.laneatt.keep_threshold}, "
                         f"match_tolerance={self.laneatt.match_tolerance}, "
                         f"yolo_conf={self.yolo.conf_threshold}, yolo_iou={self.yolo.iou_threshold}")
        
        
        if not (0 <= frame_number < total_frames):
            cap.release()
            raise ValueError(f"frame {frame_number} out of range (video has {total_frames} frames)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        convert = np.array(frame)
        # print(f"Get Image numpy: {convert}")
        # print(f"Get Image Shape: {convert.shape}")
        
        if not ret:
            raise RuntimeError(f"Could not decode frame {frame_number} of {self.video_path}")

        base_frame = frame.copy()
        frame = self.get_frame(frame) 
        frame = cv2.hconcat([base_frame, frame])


        
        cv2.imwrite(final_video_path, frame)
        self.logger.info(f"saved: {final_video_path}")
        print(f"Output Located: {final_video_path}")
        # return steering, ego_vehicle
