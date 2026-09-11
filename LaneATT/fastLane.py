
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
import matplotlib.pyplot as plt
from pathlib import Path
from lib.config import Config
import cv2
import torch

import warnings
warnings.filterwarnings('ignore')
    
    # print(evaluation)
import numpy as np
import sys

class VideoInference():
    """Hub of the pipeline: owns the video loop, run folders, logging and drawing.
    model_type picks the lane model: "laneATT" -> LaneATT.py (LaneATTInference),
    "laneNet" -> lanenet_infer.py (LaneNetInference); object detection in
    yolo.py (YoloInference); parameters flow from here into those."""

    def __init__(self, model_type = "laneATT", model_archiecture = None, model_path = None, hnet_path = None, video_path = None, output_folder = None, view = True, 
                 frame_limit = 10000000000, device = torch.device("cuda:0"), conf_threshold = 0.5,  nms_thres = 50, nms_topk = 4,
                 keep_threshold = 0.3, match_tolerance = 0.05, yolo_path = None, yolo_conf = 0.5, yolo_iou = 0.15):


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
        self.laneatt = LaneATTInference(self.model_archiecture,  self.model_path, device=self.device,
                                            conf_threshold=self.conf_threshold, nms_thres=self.nms_thres,
                                            nms_topk=self.nms_topk, keep_threshold=self.keep_threshold,
                                            match_tolerance=self.match_tolerance, )
    def update_yolo(self):
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
        # Swap in a LaneATT checkpoint and make LaneATT the active lane model.
        if self.laneatt is None:
            self.laneatt = LaneATTInference(model_archiecture, model_path, device=self.device)
        else:
            self.laneatt.model_archiecture = model_archiecture
            self.laneatt.load_model(model_path)
        self.model_type = "laneATT"

    def set_lanenet(self, model_path, hnet_path=None):
        # Load LaneNet (+ optional H-Net) and makepath it the active lane model.
        self.lanenet = LaneNetInference(model_path, hnet_path=hnet_path, device=self.device)
        self.model_type = "laneNet"

    def speed_eval(self, speed):
        # First Base Speed on if edges are found 
    
        return
    
    def get_frame(self, frame, split):

        t0 = time.perf_counter()
        evaluation = self.laneatt.frame_eval(frame)
        lane_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        yolo_results = self.yolo.infer(frame)
        yolo_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        # depth_results = self.depth.infer(frame)
        # depth_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts_all = self.laneatt.lanes_to_px(evaluation, frame.shape[1], frame.shape[0])
        # for pts in pts_all:
        #     for p0, p1 in zip(pts[:-1], pts[1:]):
        #         cv2.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 3)
                
  
        if split == "base":
            left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes(frame.shape[1], pts_all)
        elif split == "new":
            left_points, right_points, mid_points, synthesized = self.laneatt.get_ego_lanes2(frame.shape[1], pts_all)
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
                
        frame = self.yolo.draw(frame, yolo_results)

        return frame

    def video_eval(self):
        if self.video_path is None or not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video path does not exist: {self.video_path}")
        
        print(f"Video Selected: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        out_stream = None
        # folder_path = "frame_output"
        # if (self.output_folder != None):
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
        self.logger.info(f"devices: LaneATT {self.device}, "
                         f"YOLO {self.yolo.model.device}")
        self.logger.info(f"model checkpoint: {self.laneatt.model_path}")
        self.logger.info(f"params: conf_threshold={self.laneatt.conf_threshold}, "
                         f"nms_thres={self.laneatt.nms_thres}, nms_topk={self.laneatt.nms_topk}, "
                         f"keep_threshold={self.laneatt.keep_threshold}, "
                         f"match_tolerance={self.laneatt.match_tolerance}, "
                         f"yolo_conf={self.yolo.conf_threshold}, yolo_iou={self.yolo.iou_threshold}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Output Located: {final_video_path}")
        out_stream = cv2.VideoWriter(str(final_video_path), fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 3, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        
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

       
        self.logger.info(f"devices: {self.device}, "
                         f"YOLO {self.yolo.model.device}")
       
        self.laneatt.reset_video_state()
        self.angle.reset_video_state()
    
        lane_time = 0.0      
        yolo_time = 0.0      
        depth_time = 0.0     

        t1 = time.time()
        
        
        frame_location = Path(folder_path) / Path("frames")
        frame_location.mkdir(parents=True, exist_ok=True)
        print(frame_location)

        while i < local_frame_local:
            ret, frame = cap.read()
            
            frame2 = frame.copy()
            base_frame = frame.copy()
            
           
            if not ret:
                break
            
            frame = self.get_frame(frame, "base")
            frame2 = self.get_frame(frame2, "new")
            frame = cv2.hconcat([base_frame, frame, frame2])
           
            if (self.output_folder != None):
               out_stream.write(frame)

            if (i) % max(1, math.floor(local_frame_local / 50)) == 0:
                n = i + 1
                
                print(f"Frame: {i}/{local_frame_local}")
            
                self.logger.info(f"Frame: {i}/{local_frame_local}, time: {str(time.time() - t1)}, "
                                    f"LaneATT: {lane_time:.1f}s ({1000 * lane_time / n:.0f} ms/frame), "
                                    f"YOLO: {yolo_time:.1f}s ({1000 * yolo_time / n:.0f} ms/frame), "
                                    f"Depth: {depth_time:.1f}s ({1000 * depth_time / n:.0f} ms/frame)")

            i += 1
        cap.release()
        if (self.output_folder != None):
            out_stream.release()
  
   
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
        base_frame = frame.copy()
        cap.release()
        convert = np.array(frame)
        # print(f"Get Image numpy: {convert}")
        # print(f"Get Image Shape: {convert.shape}")
        
        if not ret:
            raise RuntimeError(f"Could not decode frame {frame_number} of {self.video_path}")

        frame = self.get_frame(frame) 
        frame = cv2.hconcat([base_frame, frame])


        
        cv2.imwrite(final_video_path, frame)
        self.logger.info(f"saved: {final_video_path}")
        print(f"Output Located: {final_video_path}")
        # return steering, ego_vehicle


def image_inference(MODELS, files, frame_limit = 1000, output_path = Path("video_output_4"), yolo_conf = 0.2, nms_thres = 50, nms_topk=4, conf_threshold = 0.5, keep_threshold = 0.3):
    video = None
    print(f"len: {len(MODELS)}")
    for config_path, path_model, path_yolo in MODELS:
        if not (Path(config_path).exists() and Path(path_model).exists()):
            print(f"SKIPPING {path_model}: config or checkpoint not found")
            continue
        if not Path(path_yolo).exists():
            print(f"SKIPPING {path_yolo}: YOLO checkpoint not found")
            continue

        cfg = Config(config_path)
        name = Path(path_model).stem
        model_name = Path(path_model).parent.parent.name
        output_folder = Path(output_path) / model_name / name
        print(f"=== {model_name}/{name} -> {output_folder} ===")

        if video is None:
            nms_thres = 50
            nms_topk = 4
            conf_threshold = 0.3
            keep_threshold = 0.3
            video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = frame_limit, 
                                   video_path = str(files[0]), view = True, output_folder = output_folder, device = device,
                                   yolo_path = path_yolo, yolo_conf = yolo_conf, nms_thres = nms_thres, nms_topk = nms_topk, conf_threshold = conf_threshold, keep_threshold = keep_threshold)
        else:
            # Same pipeline object: swap the LaneATT model in place, keep YOLO loaded.
            # There is no set_yolo(): every entry after the first keeps the FIRST
            # entry's YOLO model, even if its tuple names a different checkpoint.
            video.set_model(cfg.get_model(), path_model)
            video.set_output_folder(output_folder)

        if isinstance(files, str):
            video.set_video_path(files)
            video.image_eval(frame_limit)
        else:
            for i in files:
                print(f"for loop statement. We are testing the video: {str(i)}")
                video.set_video_path(str(i))
                video.image_eval(frame_limit)


def video_inference(MODELS, files, frame_limit = 1000, output_root = Path("video_output_4"), yolo_conf = 0.2, nms_thres = 50, nms_topk=4, conf_threshold = 0.5, keep_threshold = 0.3):
    video = None
    model_times = []   # (label, seconds) per model, printed at the end
    for config_path, path_model, path_yolo in MODELS:
        print('asdasdsadasdasdsadsa')
        if not (Path(config_path).exists() and Path(path_model).exists()):
            print(f"SKIPPING {path_model}: config or checkpoint not found")
            continue
        if not Path(path_yolo).exists():
            print(f"SKIPPING {path_yolo}: YOLO checkpoint not found")
            continue

        cfg = Config(config_path)
        name = Path(path_model).stem
        model_name = Path(path_model).parent.parent.name
        output_folder = Path(output_root) / model_name / name
        print(f"=== {model_name}/{name} -> {output_folder} ===")
    
        if video is None:
            nms_thres = 50
            nms_topk = 4
            conf_threshold = 0.3
            keep_threshold = 0.3
            video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = frame_limit, 
                                   video_path = str(files[0]), view = True, output_folder = output_folder, device = device,
                                   yolo_path = path_yolo, yolo_conf = yolo_conf, nms_thres = nms_thres, nms_topk = nms_topk, conf_threshold = conf_threshold, keep_threshold = keep_threshold)
        else:
            # Same pipeline object: swap the LaneATT model in place, keep YOLO loaded.
            # There is no set_yolo(): every entry after the first keeps the FIRST
            # entry's YOLO model, even if its tuple names a different checkpoint.
            video.set_model(cfg.get_model(), path_model)
            video.set_output_folder(output_folder)

        t_model = time.perf_counter()
        print(f"len: {len(files)}, files: {files}-------------------------------------------")
        if isinstance(files, str):
            video.set_video_path(files)
            video.video_eval()
        else:
            for i in files:
                print(f"for loop statement. We are testing the video: {str(i)}")
                video.set_video_path(str(i))
                video.video_eval()
        
            
        model_times.append((f"{model_name}/{name}", time.perf_counter() - t_model))

    print("\n=== time per model (all videos) ===")
    for label, seconds in model_times:
        print(f"{label}: {seconds:.1f} s ({seconds / 60:.1f} min)")

        

def build_models(args):
    """MODELSED is the default; --config/--laneatt/--yolo each override that field
    for every run (MODELSED entries share one LaneATT config+checkpoint, so the
    first entry fills in whatever isn't overridden)."""
    if args.config is None and args.laneatt is None and args.yolo is None:
        return MODELSED
    config_path = args.config or MODELSED[0][0]
    laneatt_path = args.laneatt or MODELSED[0][1]
    yolos = args.yolo if args.yolo is not None else [m[2] for m in MODELSED]
    return [(config_path, laneatt_path, str(y)) for y in yolos]


video_example = "video_input/IMG_5106.mp4"

if Path(video_example).exists():
    print("Video Exists")
else:
    print("Video Not exists")

output_folder = Path("video_inference")

MODELSED = ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "onnxmodels/YolloS/yolo11s_coco4.pt")

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print(f"build Model: {[MODELSED]}")
print(f"args.videos: {video_example}")
print(f"args.frame_limit: {500}")
print(f"args.output_dir: {output_folder}")
print(f"args.yolo_conf: {0.7}")




nms_thres = 50
nms_topk=8
conf_threshold = 0.5
keep_threshold = 0.3
frame_limit = 99999


video_inference([MODELSED], video_example, frame_limit, output_folder, 0.7, nms_thres,  nms_topk, conf_threshold, keep_threshold)


# nms_thres = 50
# nms_topk=4
# conf_threshold = 0.3
# keep_threshold = 0.3

# video_inference([MODELSED], video_example, 999999, output_folder, 0.7, nms_thres,  nms_topk, conf_threshold, keep_threshold)


# nms_thres = 50
# nms_topk=4
# conf_threshold = 0.5
# keep_threshold = 0.3

# video_inference([MODELSED], video_example, 999999, output_folder, 0.7, nms_thres,  nms_topk, conf_threshold, keep_threshold)

# output_folder = Path("image_inference")




# image_inference([MODELSED], video_example, 1202, output_path = output_folder, yolo_conf = 0.7)


# video_inference(build_models(args), args.videos, args.frame_limit, output_root=args.output_dir, yolo_conf=args.yolo_conf)
