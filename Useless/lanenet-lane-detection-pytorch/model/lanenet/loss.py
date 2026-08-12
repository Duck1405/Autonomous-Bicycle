# coding: utf-8
"""
This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
"""

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.functional import F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''
    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device = DEVICE):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        target = target.to(device=input.device, dtype=torch.long)
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001,max=0.999999)
        target_onehot = torch.zeros(
            (target.size(0), self.n_class, target.size(1), target.size(2)),
            dtype=input.dtype,
            device=input.device,
        )
        loss = 0
        for i in range(self.n_class):
            target_onehot[:,i,...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:,i,...]) ** self.gamma * target_onehot[:,i,...] * torch.log(pt[:,i,...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss

class BoundedInverseWeightedCrossEntropy(nn.Module):
    """Cross-entropy with ENet-style bounded inverse class weighting.

    The LaneNet paper (Neven et al., 2018, Sec. II-A) trains the binary
    segmentation branch with "standard cross-entropy ... with bounded inverse
    class weighting, as described in [ENet]". ENet (Paszke et al., 2016) uses

        w_class = 1 / ln(c + p_class)

    with c = 1.02, which bounds the per-class weights to roughly [1, 50].
    Here the class probabilities p_class are estimated from the current batch
    (a self-contained approximation of the dataset-level statistics).
    """

    def __init__(self, n_class=2, c=1.02):
        super().__init__()
        self.n_class = n_class
        self.c = c

    def forward(self, logits, target):
        target = target.to(device=logits.device, dtype=torch.long)
        counts = torch.bincount(target.reshape(-1), minlength=self.n_class).to(logits.dtype)
        prob = counts / counts.sum().clamp(min=1.0)
        weight = 1.0 / torch.log(self.c + prob)
        return nn.functional.cross_entropy(logits, target, weight=weight)

class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=False, size_average=True):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):

        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size, embed_dim, H, W = embedding.shape
        embedding = embedding.reshape(batch_size, embed_dim, H*W)
        seg_gt = seg_gt.reshape(batch_size, H*W)

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H*W)
            seg_gt_b = seg_gt[b]  # (H*W)

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)

                if not seg_mask_i.any():
                    continue
                
                embedding_i = embedding_b * seg_mask_i
                mean_i = torch.sum(embedding_i, dim=1) / torch.sum(seg_mask_i)
                centroid_mean.append(mean_i)
                # ---------- var_loss -------------
                var_loss = var_loss + torch.sum(F.relu(
                    torch.norm(embedding_i[:,seg_mask_i] - mean_i.reshape(embed_dim, 1), dim=0) - self.delta_var) ** 2) / torch.sum(seg_mask_i) / num_lanes
            if len(centroid_mean) == 0:
                continue
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)

                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)   # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist

                # Eq. (1): normalize by the number of ordered cluster pairs,
                # C*(C-1). Summing over the full (symmetric) distance matrix
                # already counts each pair twice, which matches that denominator.
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                        num_lanes * (num_lanes - 1))

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        return var_loss, dist_loss, reg_loss
