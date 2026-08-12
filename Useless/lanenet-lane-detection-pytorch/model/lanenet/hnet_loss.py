"""H-Net curve-fitting loss and lane-fitting utilities.

Implements the loss from "Towards End-to-End Lane Detection: an Instance
Segmentation Approach" (Neven et al., 2018), Sec. II-B. There is no ground-truth
homography; instead H-Net is supervised by how well a low-order polynomial fits
the lane *after* its predicted perspective transform:

    1. transform the lane points with H               p' = H p
    2. least-squares fit  x' = f(y')  (degree `order`)
    3. evaluate the fit at each y'                     x'* = f(y'_i)
    4. reproject the fitted points with H^-1           p* = H^-1 [x'*, y', 1]
    5. loss = mean( (x* - x)^2 )  measured in the original image frame

Every step is differentiable (the fit is a closed-form linear solve), so the
gradient flows back into H-Net's weights.

Coordinate convention: all point coordinates are expected to be normalized to
roughly [0, 1] (e.g. pixel_x / image_width, pixel_y / image_height). Normalizing
keeps the Vandermonde solve well-conditioned and makes the transform independent
of the working resolution, so training (TuSimple GT points) and inference
(LaneNet cluster pixels) share the same frame.
"""

import torch
import torch.nn as nn

from model.lanenet.backbone.H_Net import build_H


def _vandermonde(y, order):
    # (M, order+1) with columns [1, y, y^2, ..., y^order]
    return torch.stack([y ** p for p in range(order + 1)], dim=1)


def transform_points(H, pts):
    """Apply a 3x3 homography to (M, 2) points [x, y]; returns (M, 2) [x', y']
    after perspective division."""
    ones = torch.ones(pts.shape[0], dtype=pts.dtype, device=pts.device)
    P = torch.stack([pts[:, 0], pts[:, 1], ones], dim=1)   # (M, 3)
    Pp = P @ H.t()                                         # (M, 3)
    w = Pp[:, 2]
    return torch.stack([Pp[:, 0] / w, Pp[:, 1] / w], dim=1)


def fit_polynomial(y, x, order, ridge=1e-6):
    """Closed-form least-squares fit of x = f(y), degree `order`.
    Solves the (ridge-regularized) normal equations (V^T V) w = V^T x."""
    V = _vandermonde(y, order)
    A = V.t() @ V + ridge * torch.eye(order + 1, dtype=V.dtype, device=V.device)
    b = V.t() @ x
    return torch.linalg.solve(A, b)                        # (order+1,)


def _reproject(H, x_t, y_t):
    """Map points [x_t, y_t] from transformed space back to image space with
    H^-1; returns (M, 2). Uses solve(H, .) instead of an explicit inverse."""
    ones = torch.ones_like(y_t)
    P = torch.stack([x_t, y_t, ones], dim=1)               # (M, 3)
    Pstar = torch.linalg.solve(H, P.t()).t()               # (Hinv @ P^T)^T
    return torch.stack([Pstar[:, 0] / Pstar[:, 2],
                        Pstar[:, 1] / Pstar[:, 2]], dim=1)


def hnet_curve_fit_loss(params, batch_lanes, order=3, ridge=1e-6, min_points=None):
    """params: (N, 6) raw H-Net output.
    batch_lanes: list of length N; batch_lanes[i] is a list of (M_k, 2) tensors
    holding the normalized [x, y] points of each ground-truth lane in image i.
    """
    if min_points is None:
        min_points = order + 1

    H_all = build_H(params)                                # (N, 3, 3)
    total = torch.zeros((), dtype=params.dtype, device=params.device)
    count = 0

    for i, lanes in enumerate(batch_lanes):
        H = H_all[i]
        for pts in lanes:
            if pts.shape[0] < min_points:
                continue
            pts = pts.to(device=params.device, dtype=params.dtype)
            x = pts[:, 0]
            tp = transform_points(H, pts)
            xp, yp = tp[:, 0], tp[:, 1]
            w = fit_polynomial(yp, xp, order, ridge)
            x_fit = _vandermonde(yp, order) @ w
            x_star = _reproject(H, x_fit, yp)[:, 0]
            total = total + torch.sum((x_star - x) ** 2)
            count += pts.shape[0]

    if count == 0:
        # Keep a connection to the graph so .backward() is always valid.
        return params.sum() * 0.0
    return total / count


class HNetLoss(nn.Module):
    def __init__(self, order=3, ridge=1e-6):
        super().__init__()
        self.order = order
        self.ridge = ridge

    def forward(self, params, batch_lanes):
        return hnet_curve_fit_loss(params, batch_lanes, self.order, self.ridge)


@torch.no_grad()
def fit_lane_bev(H, points_norm, order=3, ridge=1e-6, n_samples=50):
    """Inference helper: given a 3x3 homography `H` (e.g. build_H(params)[i]) and
    one lane's normalized [x, y] points, fit the curve in transformed space and
    return `n_samples` curve points reprojected to the normalized image frame,
    shape (n_samples, 2). Returns None if there are too few points to fit.

    Multiply the result by (image_width, image_height) to get pixel coordinates.
    """
    if points_norm.shape[0] < order + 1:
        return None
    points_norm = points_norm.to(dtype=H.dtype, device=H.device)
    tp = transform_points(H, points_norm)
    xp, yp = tp[:, 0], tp[:, 1]
    w = fit_polynomial(yp, xp, order, ridge)
    y_samp = torch.linspace(float(yp.min()), float(yp.max()), n_samples,
                            dtype=H.dtype, device=H.device)
    x_samp = _vandermonde(y_samp, order) @ w
    return _reproject(H, x_samp, y_samp)
