import argparse
import time
from pathlib import Path

import cv2
import torch

from utils.utils import (
    AverageMeter,
    driving_area_mask,
    increment_path,
    lane_line_mask,
    letterbox,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    select_device,
    show_seg_result,
    split_for_trace_model,
    time_synchronized,
    xyxy2xywh,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='camera index or video device path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3, cpu, or leave blank for auto')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save detections to *.txt')
    parser.add_argument('--project', default='runs/webcam', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-video', action='store_true', help='save annotated webcam video')
    parser.add_argument('--view-width', type=int, default=1280, help='capture/display width')
    parser.add_argument('--view-height', type=int, default=720, help='capture/display height')
    return parser


def parse_source(source):
    source = str(source).strip()
    if source.isdigit():
        return int(source)
    return source


def preprocess_frame(frame, img_size, stride):
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy())
    return img, frame


def main(opt):
    source = parse_source(opt.source)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    if opt.save_txt or opt.save_video:
        save_dir.mkdir(parents=True, exist_ok=True)
    if opt.save_txt:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)

    stride = 32
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = torch.jit.load(opt.weights, map_location=device).to(device)
    if half:
        model.half()
    model.eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera source: {opt.source}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.view_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.view_height)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))

    video_writer = None
    if opt.save_video:
        output_path = str(save_dir / 'webcam.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (opt.view_width, opt.view_height),
        )

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    waste_time = AverageMeter()
    frame_idx = 0
    window_name = 'YOLOPv2 Webcam'
    print("Starting webcam inference. Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                print('Camera frame read failed, stopping.')
                break

            img, im0 = preprocess_frame(frame, opt.img_size, stride)
            img = img.to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = model(img)
            t2 = time_synchronized()

            tw1 = time_synchronized()
            pred = split_for_trace_model(pred, anchor_grid)
            tw2 = time_synchronized()

            t3 = time_synchronized()
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )
            t4 = time_synchronized()

            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            label_path = save_dir / 'labels' / f'frame_{frame_idx:06d}.txt'
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if opt.save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(label_path, 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        plot_one_box(xyxy, im0, line_thickness=3)

            inf_time.update(t2 - t1, img.size(0))
            nms_time.update(t4 - t3, img.size(0))
            waste_time.update(tw2 - tw1, img.size(0))

            total_time = max(t4 - t1, 1e-6)
            fps = 1.0 / total_time
            cv2.putText(
                im0,
                f'FPS: {fps:.1f} | Device: {device.type.upper()}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, im0)
            if video_writer is not None:
                video_writer.write(im0)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f'Saved webcam video to: {save_dir / "webcam.mp4"}')
    cv2.destroyAllWindows()
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)   waste : (%.4fs/frame)' % (inf_time.avg, nms_time.avg, waste_time.avg))


if __name__ == '__main__':
    options = make_parser().parse_args()
    print(options)
    main(options)
