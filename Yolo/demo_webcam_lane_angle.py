import argparse
from pathlib import Path

import cv2
import numpy as np
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
    parser.add_argument('--project', default='runs/lane_angle', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-video', action='store_true', help='save annotated webcam video')
    parser.add_argument('--view-width', type=int, default=1280, help='capture/display width')
    parser.add_argument('--view-height', type=int, default=720, help='capture/display height')
    parser.add_argument('--smooth-alpha', type=float, default=0.18, help='EMA factor for angle smoothing')
    parser.add_argument('--max-angle', type=float, default=45.0, help='max steering angle shown on the overlay')
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
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    return img, frame


def build_focus_roi(shape):
    height, width = shape
    roi = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[
            (int(width * 0.04), height - 1),
            (int(width * 0.28), int(height * 0.56)),
            (int(width * 0.72), int(height * 0.56)),
            (int(width * 0.96), height - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, polygon, 255)
    return roi, polygon


def prepare_lane_mask(lane_mask, drive_mask):
    lane = (lane_mask > 0).astype(np.uint8) * 255
    roi, polygon = build_focus_roi(lane.shape)

    lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    lane = cv2.dilate(lane, np.ones((5, 5), dtype=np.uint8), iterations=1)
    lane = cv2.bitwise_and(lane, lane, mask=roi)
    return lane, polygon


def choose_lane_point(candidates, reference, side, max_jump):
    if candidates.size == 0:
        return None

    if reference is None:
        return int(candidates.max() if side == 'left' else candidates.min())

    distances = np.abs(candidates - reference)
    best_index = int(np.argmin(distances))
    if distances[best_index] > max_jump:
        return None
    return int(candidates[best_index])


def find_initial_bounds(lane_binary, center_x, y_start, y_stop, step, min_lane_width, max_lane_width):
    for y in range(y_start, y_stop, -step):
        xs = np.flatnonzero(lane_binary[y] > 0)
        if xs.size == 0:
            continue

        left_candidates = xs[xs < center_x]
        right_candidates = xs[xs > center_x]
        if left_candidates.size == 0 or right_candidates.size == 0:
            continue

        left_x = int(left_candidates.max())
        right_x = int(right_candidates.min())
        width = right_x - left_x
        if min_lane_width <= width <= max_lane_width:
            return y, left_x, right_x

    return None, None, None


def extract_ego_lane_points(lane_binary):
    height, width = lane_binary.shape
    center_x = width // 2
    y_bottom = int(height * 0.94)
    y_top = int(height * 0.56)
    row_step = 6
    min_lane_width = int(width * 0.10)
    max_lane_width = int(width * 0.52)
    max_jump = int(width * 0.10)

    start_y, left_x, right_x = find_initial_bounds(
        lane_binary,
        center_x,
        y_bottom,
        y_top,
        row_step,
        min_lane_width,
        max_lane_width,
    )
    if start_y is None:
        return None

    lane_width = right_x - left_x
    left_points = []
    right_points = []
    center_points = []

    for y in range(start_y, y_top, -row_step):
        xs = np.flatnonzero(lane_binary[y] > 0)
        if xs.size == 0:
            continue

        left_candidates = xs[xs < center_x]
        right_candidates = xs[xs > center_x]
        left_candidate = choose_lane_point(left_candidates, left_x, 'left', max_jump)
        right_candidate = choose_lane_point(right_candidates, right_x, 'right', max_jump)

        if left_candidate is None or right_candidate is None:
            continue

        width_now = right_candidate - left_candidate
        if not (min_lane_width <= width_now <= max_lane_width):
            continue

        if abs(width_now - lane_width) > int(width * 0.12):
            continue

        left_x = left_candidate
        right_x = right_candidate
        lane_width = int(0.8 * lane_width + 0.2 * width_now)

        left_points.append((left_x, y))
        right_points.append((right_x, y))
        center_points.append(((left_x + right_x) // 2, y))

    if len(center_points) < 8:
        return None

    return {
        'left': left_points,
        'right': right_points,
        'center': center_points,
        'image_center_x': center_x,
    }


def fit_centerline(center_points, frame_shape):
    points = np.array(center_points, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]

    if len(points) >= 12:
        coeffs = np.polyfit(ys, xs, deg=2)
        fit = np.poly1d(coeffs)
    else:
        coeffs = np.polyfit(ys, xs, deg=1)
        fit = np.poly1d(coeffs)

    height = frame_shape[0]
    y_near = min(int(height * 0.90), int(ys.max()))
    y_far = max(int(height * 0.62), int(ys.min()))
    if y_near <= y_far:
        return None

    x_near = float(fit(y_near))
    x_far = float(fit(y_far))
    dx = x_far - x_near
    dy = float(y_near - y_far)
    angle_deg = float(np.degrees(np.arctan2(dx, dy)))

    sample_ys = np.linspace(y_near, y_far, num=20)
    fitted_points = [(int(fit(y)), int(y)) for y in sample_ys]
    return {
        'angle_deg': angle_deg,
        'x_near': x_near,
        'x_far': x_far,
        'y_near': y_near,
        'y_far': y_far,
        'fitted_points': fitted_points,
    }


def smooth_angle(current_angle, previous_angle, alpha):
    if current_angle is None:
        return previous_angle
    if previous_angle is None:
        return current_angle
    return (1.0 - alpha) * previous_angle + alpha * current_angle


def format_direction(angle_deg):
    if angle_deg is None:
        return 'Searching', 'Lane angle unavailable', (180, 180, 180)

    if abs(angle_deg) < 2.0:
        return 'Straight', 'Hold current heading', (80, 220, 80)

    if angle_deg > 0:
        return 'Right', f'Steer right {abs(angle_deg):.1f} deg', (0, 210, 255)

    return 'Left', f'Steer left {abs(angle_deg):.1f} deg', (0, 165, 255)


def draw_heading_overlay(frame, lane_data, fit_data, angle_deg, max_angle):
    overlay = frame.copy()
    height, width = frame.shape[:2]

    if lane_data is not None:
        for x, y in lane_data['left']:
            cv2.circle(overlay, (x, y), 2, (255, 180, 0), -1)
        for x, y in lane_data['right']:
            cv2.circle(overlay, (x, y), 2, (255, 180, 0), -1)

    if fit_data is not None:
        cv2.polylines(
            overlay,
            [np.array(fit_data['fitted_points'], dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3,
        )
        cv2.line(
            overlay,
            (width // 2, fit_data['y_near']),
            (width // 2, fit_data['y_far']),
            (255, 255, 255),
            2,
        )

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    direction_label, action_label, color = format_direction(angle_deg)
    displayed_angle = 0.0 if angle_deg is None else angle_deg
    panel_left, panel_top = 20, 20
    panel_right, panel_bottom = 360, 155
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), color, 2)
    cv2.putText(frame, 'Lane Guidance', (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Heading Error: {displayed_angle:+.1f} deg', (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'Direction: {direction_label}', (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, action_label, (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)

    gauge_left, gauge_top = 30, height - 65
    gauge_width, gauge_height = 240, 24
    cv2.rectangle(frame, (gauge_left, gauge_top), (gauge_left + gauge_width, gauge_top + gauge_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (gauge_left, gauge_top), (gauge_left + gauge_width, gauge_top + gauge_height), (220, 220, 220), 1)
    center_x = gauge_left + gauge_width // 2
    cv2.line(frame, (center_x, gauge_top - 6), (center_x, gauge_top + gauge_height + 6), (255, 255, 255), 2)

    clamped = float(np.clip(displayed_angle, -max_angle, max_angle))
    offset = int((clamped / max_angle) * (gauge_width // 2 - 8))
    cv2.rectangle(
        frame,
        (center_x, gauge_top + 4),
        (center_x + offset, gauge_top + gauge_height - 4),
        color,
        -1,
    )
    cv2.putText(frame, 'Left', (gauge_left, gauge_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Right', (gauge_left + gauge_width - 45, gauge_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def main(opt):
    source = parse_source(opt.source)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    if opt.save_video:
        save_dir.mkdir(parents=True, exist_ok=True)

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    video_writer = None
    if opt.save_video:
        output_path = str(save_dir / 'lane_angle_webcam.mp4')
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (opt.view_width, opt.view_height),
        )

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    angle_time = AverageMeter()
    smoothed_angle = None
    window_name = 'YOLOPv2 Lane Angle'
    print("Starting lane-angle webcam inference. Press 'q' to quit.")

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

            pred = split_for_trace_model(pred, anchor_grid)
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

            t5 = time_synchronized()
            lane_binary, roi_polygon = prepare_lane_mask(ll_seg_mask, da_seg_mask)
            lane_data = extract_ego_lane_points(lane_binary)
            fit_data = None if lane_data is None else fit_centerline(lane_data['center'], im0.shape)
            raw_angle = None if fit_data is None else fit_data['angle_deg']
            smoothed_angle = smooth_angle(raw_angle, smoothed_angle, opt.smooth_alpha)
            t6 = time_synchronized()

            cv2.polylines(im0, [roi_polygon], isClosed=True, color=(60, 60, 60), thickness=2)
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, im0, line_thickness=3)

            draw_heading_overlay(im0, lane_data, fit_data, smoothed_angle, opt.max_angle)
            cv2.putText(
                im0,
                f'Inference: {(t2 - t1) * 1000:.0f} ms',
                (opt.view_width - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            inf_time.update(t2 - t1, img.size(0))
            nms_time.update(t4 - t3, img.size(0))
            angle_time.update(t6 - t5, img.size(0))

            cv2.imshow(window_name, im0)
            if video_writer is not None:
                video_writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f'Saved webcam video to: {save_dir / "lane_angle_webcam.mp4"}')
    cv2.destroyAllWindows()
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)   angle : (%.4fs/frame)' % (inf_time.avg, nms_time.avg, angle_time.avg))


if __name__ == '__main__':
    options = make_parser().parse_args()
    print(options)
    main(options)
