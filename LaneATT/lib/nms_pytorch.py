"""Pure-PyTorch fallback for the CUDA lane-NMS extension in `lib/nms`.

This reimplements `nms(boxes, scores, overlap, top_k)` from `nms/src/nms_kernel.cu`
using only standard torch ops, so it runs on CPU / MPS (e.g. Apple Silicon) where
the CUDA extension cannot be built or run.

A proposal is a lane encoded as `[score0, score1, start_y, start_x, length, x_0 ... x_{S-1}]`
(`5 + n_offsets` values). "Overlap" between two lanes is the *mean absolute horizontal
distance* between their x-offsets over the vertical span where both lanes exist; two
lanes are considered duplicates (and the lower-scored one suppressed) when that mean
distance is below `overlap`. This matches the `devIoU` logic in the CUDA kernel.

Returns `(keep, num_to_keep, parent_object_index)` with the same semantics as the CUDA
op so it is a drop-in replacement: `keep` is length `N` with the kept original indices
(score-sorted) first and the remainder zero-filled.
"""
import torch


def nms(boxes, scores, overlap, top_k):
    device = boxes.device
    n = boxes.shape[0]

    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, torch.tensor(0, dtype=torch.long, device=device), empty

    n_offsets = boxes.shape[1] - 5
    n_strips = n_offsets - 1

    # Process highest-confidence proposals first (the CUDA op sorts internally).
    order = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[order]

    # Fast path: a non-positive threshold can never suppress anything (mean distance
    # is >= 0, and the test is strict `< overlap`), so just take the top-k by score.
    # This is the training case (`nms_thres=0`) and avoids the O(N^2) work entirely.
    if overlap <= 0:
        num = min(n, int(top_k))
        keep = torch.zeros(n, dtype=torch.long, device=device)
        keep[:num] = order[:num]
        parent = torch.zeros(n, dtype=torch.long, device=device)
        parent[order[:num]] = torch.arange(1, num + 1, device=device)
        return keep, torch.tensor(num, dtype=torch.long, device=device), parent

    # Per-proposal vertical extent (in offset-index space), matching the CUDA `devIoU`:
    #   start = round(start_y * n_strips);  end = round(start + length - 1)
    starts = (sorted_boxes[:, 2] * n_strips + 0.5).long()
    lengths = sorted_boxes[:, 4]
    ends = (starts.float() + lengths - 1 + 0.5).long() - (lengths - 1 < 0).long()
    ends = torch.clamp(ends, max=n_offsets - 1)
    xs = sorted_boxes[:, 5:5 + n_offsets]  # [N, n_offsets]

    offset_idx = torch.arange(n_offsets, device=device)  # [n_offsets]

    removed = torch.zeros(n, dtype=torch.bool, device=device)
    parent = torch.zeros(n, dtype=torch.long, device=device)
    keep = torch.zeros(n, dtype=torch.long, device=device)
    num = 0

    for i in range(n):
        if removed[i]:
            continue

        keep[num] = order[i]
        num += 1
        parent[order[i]] = num

        rest = slice(i + 1, n)
        # Vertical span shared by proposal i and every later proposal.
        s = torch.maximum(starts[i], starts[rest])           # [M]
        e = torch.minimum(ends[i], ends[rest])               # [M]
        valid = (offset_idx[None, :] >= s[:, None]) & (offset_idx[None, :] <= e[:, None])  # [M, n_offsets]

        diff = (xs[i][None, :] - xs[rest]).abs() * valid     # [M, n_offsets]
        counts = valid.sum(dim=1)                            # [M]
        mean_dist = diff.sum(dim=1) / counts.clamp(min=1)

        # Suppress later proposals that overlap vertically, are close enough, and
        # haven't already been removed by an earlier kept proposal.
        suppress = (counts > 0) & (mean_dist < overlap) & (~removed[rest])
        sup_abs = torch.nonzero(suppress, as_tuple=True)[0] + (i + 1)
        removed[sup_abs] = True
        parent[order[sup_abs]] = num

        if num == int(top_k):
            break

    return keep, torch.tensor(num, dtype=torch.long, device=device), parent
