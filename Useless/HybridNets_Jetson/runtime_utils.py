from __future__ import annotations

import cv2
import numpy as np


def letterbox(combination, new_shape=(384, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    img, seg = combination
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        ratio = min(ratio, 1.0)

    resize_ratio = (ratio, ratio)
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        resize_ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if seg:
            for seg_class in seg:
                seg[seg_class] = cv2.resize(seg[seg_class], new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if seg:
        for seg_class in seg:
            seg[seg_class] = cv2.copyMakeBorder(
                seg[seg_class],
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

    return (img, seg), resize_ratio, (dw, dh)
