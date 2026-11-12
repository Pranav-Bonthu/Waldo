"""Scoring detections against ground truth."""

import numpy as np


def iou(a, b):
    """Intersection over Union of two [x, y, w, h] boxes.

    IoU = overlap area / area covered by either box. 1.0 is a perfect match,
    0.0 is no overlap. A detection counts as correct when IoU >= 0.5.
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax0 + aw, bx0 + bw)
    iy1 = min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def hit_rank(candidates, truth, iou_threshold=0.5):
    """1-based rank of the first correct candidate, or None if none are correct."""
    for i, c in enumerate(candidates):
        if iou(c["box"], truth) >= iou_threshold:
            return i + 1
    return None


def best_threshold(pos_scores, neg_scores):
    """Pick a 'not found' score cutoff that best separates the two groups.

    Tries a cutoff between every pair of sorted scores and keeps the one with
    the highest balanced accuracy (average of: fraction of positives above,
    fraction of negatives below). Returns (threshold, neg_rejected_fraction).
    """
    if not neg_scores:
        return None, None
    pos = np.array(pos_scores, dtype=np.float64)
    neg = np.array(neg_scores, dtype=np.float64)
    values = np.sort(np.concatenate([pos, neg]))
    cuts = np.concatenate([[values[0] - 1.0], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1.0]])
    best_cut = cuts[0]
    best_bal = -1.0
    for cut in cuts:
        bal = 0.5 * (np.mean(pos >= cut) + np.mean(neg < cut))
        if bal > best_bal:
            best_bal = bal
            best_cut = cut
    return float(best_cut), float(np.mean(neg < best_cut))
