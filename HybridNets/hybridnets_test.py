import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from utils.segmentation import segmentation_logits_to_probabilities, segmentation_probabilities_to_predictions, \
    resize_segmentation_probabilities
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from torch.nn import functional as F



def _select_primary_corridor(corridor_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corridor_mask, connectivity=8)
    if num_labels <= 2:
        return corridor_mask

    height, width = corridor_mask.shape
    anchor_x = width // 2
    anchor_y = min(height - 1, int(height * 0.92))

    anchor_label = labels[anchor_y, anchor_x]
    if anchor_label > 0:
        return (labels == anchor_label).astype(np.uint8) * 255

    best_label = 0
    best_score = float("-inf")
    image_area = float(height * width)
    for label_index in range(1, num_labels):
        x, y, component_width, component_height, area = stats[label_index]
        centroid_x, _ = centroids[label_index]
        bottom_ratio = float(y + component_height) / max(height, 1)
        center_distance = abs(float(centroid_x) - anchor_x) / max(width, 1)
        area_ratio = float(area) / max(image_area, 1.0)
        score = area_ratio * 2.0 + bottom_ratio - center_distance
        if score > best_score:
            best_score = score
            best_label = label_index

    if best_label <= 0:
        return corridor_mask
    return (labels == best_label).astype(np.uint8) * 255


def _probabilities_to_original_frame(probabilities: torch.Tensor, shape, seg_mode):
    pad_h = int(shape[1][1][1])
    pad_w = int(shape[1][1][0])
    if pad_h > 0:
        probabilities = probabilities[..., pad_h:-pad_h, :]
    if pad_w > 0:
        probabilities = probabilities[..., :, pad_w:-pad_w]
    original_height, original_width = shape[0]
    probabilities = resize_segmentation_probabilities(
        probabilities,
        size=(original_height, original_width),
    )
    if seg_mode != MULTILABEL_MODE:
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probabilities

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=True, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
parser.add_argument('--save_seg_confidence', type=boolean_string, default=False,
                    help="Save per-pixel segmentation probabilities, selected class, and confidence as .npz files")
parser.add_argument('--seg_confidence_dir', type=str, default=None,
                    help="Directory for per-pixel confidence logs. Defaults to <output>/seg_confidence")
parser.add_argument('--speed_test', type=boolean_string, default=False,
                    help='Measure inference latency')
args = parser.parse_args()

# /home/aman/Projects/Auto/models/hybridNet/results/checkpoints/hybridnets_epoch_025_weights.pth
# python hybridnets_test.py -w /home/aman/Projects/Auto/models/hybridNet/results/checkpoints/hybridnets_epoch_025_weights.pth --source /home/aman/Projects/Auto/test_images_jpg/ --output demo_result --imshow False --imwrite True

params = Params(f'projects/{args.project}.yml')
color_list_seg = {}
for seg_class in params.seg_list:
    # edit your color here if you wanna fix to your liking
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights
img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
# img_path = [img_path[0]]  # demo with 1 image
input_imgs = []
shapes = []
det_only_imgs = []

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales

threshold = args.conf_thresh
iou_threshold = args.iou_thresh
imshow = args.imshow
imwrite = args.imwrite
show_det = args.show_det
show_seg = args.show_seg
os.makedirs(output, exist_ok=True)
seg_confidence_dir = args.seg_confidence_dir or os.path.join(output, 'seg_confidence')
if args.save_seg_confidence:
    os.makedirs(seg_confidence_dir, exist_ok=True)

use_cuda = True
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list

color_list = standard_to_bgr(STANDARD_COLORS)
ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]
ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
print(f"FOUND {len(ori_imgs)} IMAGES")
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
model_image_size = params.model['image_size']
if isinstance(model_image_size, list):
    target_width, target_height = model_image_size
    resized_shape = (target_height, target_width)
else:
    resized_shape = (model_image_size, model_image_size)
normalize = transforms.Normalize(
    mean=params.mean, std=params.std
)


transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
for ori_img in ori_imgs:
    h0, w0 = ori_img.shape[:2]  # orig hw
    r = max(resized_shape) / max(h0, w0)  # resize image to img_size
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]

    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=False,
                                              scaleup=False)
    input_imgs.append(input_img)
    # cv2.imwrite('input.jpg', input_img * 255)
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling


for fi in input_imgs:
    print(fi.shape)
    t = transform(fi)
    print(t.shape)

if use_cuda:
    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
else:
    x = torch.stack([transform(fi) for fi in input_imgs], 0)

x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
# print(x.shape)
weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
#new_weight = OrderedDict((k[6:], v) for k, v in weight['model'].items())
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
model.load_state_dict(weight)

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
    if use_float16:
        model = model.half()

with torch.no_grad():
    features, regression, classification, anchors, seg = model(x)
    print("features:", [feature.size() for feature in features])
    print("regression:", regression.size()) 
    print("classification:", classification.size())
    print("anchors:", anchors.size())
    seg_probabilities = segmentation_logits_to_probabilities(seg.float(), seg_mode)

    # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
    seg_mask_list = []
    # (B, C, W, H) -> (B, W, H)
    if seg_mode == BINARY_MODE:
        print("BINARY MODE SEGMENTATION")
        seg_mask = torch.where(seg >= 0, 1, 0)
        # print(torch.count_nonzero(seg_mask))
        seg_mask.squeeze_(1)
        seg_mask_list.append(seg_mask)
    elif seg_mode == MULTICLASS_MODE:
        print("MULTICLASS MODE SEGMENTATION")
        _, seg_mask = torch.max(seg, 1)
        seg_mask_list.append(seg_mask)
    else:
        print("MULTILABEL MODE SEGMENTATION")
        seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
        # but remove background class from the list
        seg_mask_list.pop(0)
    # (B, W, H) -> (W, H)
    print("segmentation mask:", seg.size(0))
    print(seg_mask_list)
    print(len(seg_mask_list[0]))
    for i in range(seg.size(0)):
        if args.save_seg_confidence:
            frame_probabilities = _probabilities_to_original_frame(seg_probabilities[i:i + 1], shapes[i], seg_mode)
            frame_prediction, frame_confidence = segmentation_probabilities_to_predictions(frame_probabilities, seg_mode)
            frame_probabilities = frame_probabilities.squeeze(0).detach().cpu().numpy().astype(np.float32)
            frame_confidence = frame_confidence.squeeze(0).detach().cpu().numpy().astype(np.float32)
            frame_prediction = frame_prediction.squeeze(0).detach().cpu().numpy()
            if seg_mode != MULTILABEL_MODE:
                frame_prediction = frame_prediction.astype(np.uint8)
            np.savez_compressed(
                os.path.join(seg_confidence_dir, f'{i}_seg_confidence.npz'),
                probabilities=frame_probabilities,
                predicted_class=frame_prediction,
                confidence=frame_confidence,
                class_names=np.array(['background', *params.seg_list]),
                seg_mode=np.array(seg_mode),
            )
        #   print(i)
        for seg_class_index, seg_mask in enumerate(seg_mask_list):
            
            seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
            edgeMask = (seg_mask_ == 2).astype(np.uint8) * 255
            roadMask = (seg_mask_ == 1).astype(np.uint8) * 255
            print("edgeMask")
            print(edgeMask)
            print("roadMask")
            print(roadMask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            corridor_mask1 = cv2.morphologyEx(edgeMask, cv2.MORPH_CLOSE, kernel)
            corridor_mask2 = cv2.medianBlur(corridor_mask1, 5)
            corridor_mask3 = _select_primary_corridor(corridor_mask2)
                
            
            print(edgeMask.shape)
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 11.52))
            ax1.set_title("Original Image")
            ax1.imshow(input_imgs[i], origin="upper")
            ax1.axis("off")

            ax2.set_title("Edge Mask")
            ax2.imshow(
                edgeMask,
                cmap=ListedColormap(["black", "blue"]),
                origin='upper'
            )
            ax2.axis("off")

            ax3.set_title("Corridor Mask after Morphological Closing")
            ax3.imshow(
                corridor_mask1,
                cmap=ListedColormap(["black", "blue"]),
                origin='upper'
            )
            ax3.axis("off")

            ax4.set_title("Corridor Mask after Median Blur")
            ax4.imshow(
                corridor_mask2,
                cmap=ListedColormap(["black", "blue"]),
                origin='upper'
            )
            ax4.axis("off")

            ax5.set_title("Primary Corridor Mask after Connected Components")
            ax5.imshow(
                corridor_mask3,
                cmap=ListedColormap(["black", "blue"]),
                origin='upper'
            )
            ax5.axis("off")

            ax6.set_title("Original Segmentation Mask")
            ax6.imshow(
                seg_mask_,
                cmap=ListedColormap(["black", "green", "blue"]),
                vmin=0,
                vmax=2,
                origin='upper'
            )
            ax6.axis("off")

            
            plt.tight_layout()
            plt.show()

            pad_h = int(shapes[i][1][1][1]) # Shape[((original_height, original_width), ((resize_ratio_h, resize_ratio_w), pad))] getting this "resize_ratio_h"
            pad_w = int(shapes[i][1][1][0]) # Shape[((original_height, original_width), ((resize_ratio_h, resize_ratio_w), pad))] geting this  "resize_ratio_w"
            print("pad_h:", pad_h, "pad_w:", pad_w)
            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
            print("segmentation mask after removing padding:", seg_mask_.shape)
            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
            print("segmentation mask after resizing back to original shape:", seg_mask_.shape)
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
            color_seg = color_seg[..., ::-1]  # RGB -> BGR
            # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
            # prepare to show det on 2 different imgs
            # (with and without seg) -> (full and det_only)
            det_only_imgs.append(ori_imgs[i].copy())
            seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            seg_img = seg_img.astype(np.uint8)
            seg_filename = f'{output}/{i}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                           f'{output}/{i}_seg.jpg'
            if show_seg or seg_mode == MULTILABEL_MODE:
                cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    # out = postprocess(x,
    #                   anchors, regression, classification,
    #                   regressBoxes, clipBoxes,
    #                   threshold, iou_threshold)

    # for i in range(len(ori_imgs)):
    #     out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
    #     for j in range(len(out[i]['rois'])):
    #         x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
    #         obj = obj_list[out[i]['class_ids'][j]]
    #         score = float(out[i]['scores'][j])
    #         plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
    #                      color=color_list[get_index_label(obj, obj_list)])
    #         if show_det:
    #             plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
    #                          color=color_list[get_index_label(obj, obj_list)])

    #     if show_det:
    #         cv2.imwrite(f'{output}/{i}_det.jpg',  cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))

    #     if imshow:
    #         cv2.imshow('img', ori_imgs[i])
    #         cv2.waitKey(0)

    #     if imwrite:
    #         cv2.imwrite(f'{output}/{i}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))

if not args.speed_test:
    exit(0)
print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring 1 image for 10 times...')
    x = x[0, ...]
    x.unsqueeze_(0)
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    print('test2: model inferring only')
    print('inferring images for batch_size 32 for 10 times...')
    t1 = time.time()
    x = torch.cat([x] * 32, 0)
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
