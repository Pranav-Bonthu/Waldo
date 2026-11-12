"""Red/white color filter learned from the template.

Idea: Waldo's hat is red and white. A candidate box that has almost no red or
almost no white pixels is probably a false match, even if its correlation
score is high. We use this AFTER matching, to re-rank candidates.

How thresholds are learned (no hand-tuned numbers for S and V):
    1. Convert the template to HSV.
    2. Loosely pick "reddish" pixels (hue near 0 or 180, some saturation)
       and "whitish" pixels (low saturation, bright).
    3. Use the 10th percentile of their S and V values as the cutoffs, so
       the ranges fit this particular image's printing and lighting.
"""

import cv2
import numpy as np

RED_HUE_MARGIN = 10  # hue within 10 of 0 (or 180) counts as red


def _red_hue(h):
    return (h <= RED_HUE_MARGIN) | (h >= 180 - RED_HUE_MARGIN)


def learn_ranges(template_bgr):
    """Return a dict of thresholds plus the template's own red/white fractions."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)

    loose_red = _red_hue(h) & (s > 80) & (v > 80)
    loose_white = (s < 60) & (v > 170)

    ranges = {
        "red_s_min": 80,
        "red_v_min": 80,
        "white_s_max": 60,
        "white_v_min": 170,
    }
    if loose_red.sum() >= 5:
        ranges["red_s_min"] = int(np.percentile(s[loose_red], 10))
        ranges["red_v_min"] = int(np.percentile(v[loose_red], 10))
    if loose_white.sum() >= 5:
        ranges["white_s_max"] = int(np.percentile(s[loose_white], 90))
        ranges["white_v_min"] = int(np.percentile(v[loose_white], 10))

    red, white = masks(template_bgr, ranges)
    ranges["template_red_frac"] = float(red.mean())
    ranges["template_white_frac"] = float(white.mean())
    return ranges


def masks(image_bgr, ranges):
    """Boolean red and white masks for an image using learned ranges."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    red = _red_hue(h) & (s >= ranges["red_s_min"]) & (v >= ranges["red_v_min"])
    white = (s <= ranges["white_s_max"]) & (v >= ranges["white_v_min"])
    return red, white


def passes(image_bgr, box, ranges, min_ratio=0.5):
    """True if the box has at least min_ratio of the template's red AND white."""
    x, y, w, h = box
    patch = image_bgr[y:y + h, x:x + w]
    red, white = masks(patch, ranges)
    red_ok = red.mean() >= min_ratio * ranges["template_red_frac"]
    white_ok = white.mean() >= min_ratio * ranges["template_white_frac"]
    return bool(red_ok and white_ok)


def rerank(image_bgr, candidates, ranges, min_ratio=0.5):
    """Move candidates that fail the color test behind the ones that pass.

    Order inside each group is kept, so this only demotes, never invents.
    """
    keep = []
    demoted = []
    for c in candidates:
        if passes(image_bgr, c["box"], ranges, min_ratio):
            keep.append(c)
        else:
            demoted.append(c)
    return keep + demoted
