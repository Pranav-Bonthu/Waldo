"""Comparing boxes. A box is [x, y, w, h] in pixels."""


def iou(a, b):
    """Intersection over Union: overlap area / area covered by either box.

    1.0 is a perfect match, 0.0 is no overlap. A guess counts as correct
    when its IoU with the true box is at least 0.5.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def hit_rank(candidates, truth, iou_threshold=0.5):
    """1-based rank of the first correct candidate, or None if none is correct."""
    for i, c in enumerate(candidates):
        if iou(c["box"], truth) >= iou_threshold:
            return i + 1
    return None
