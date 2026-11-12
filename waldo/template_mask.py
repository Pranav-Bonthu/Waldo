"""Automatic template mask: ignore pixels that are not Waldo.

A template cut from a scene often contains things in front of or behind
Waldo. In image 1 a red spear runs down across his face and the corners are
yellow background. Correlation treats those pixels like Waldo, so the
template ends up matching other spears and other yellow areas.

A mask is an image the same size as the template: 1 = use this pixel,
0 = ignore it. OpenCV's matchTemplate accepts it directly, so the masked
pixels simply do not count in the correlation sums.

Two rules, both measured on the template itself (nothing hand-tuned per image):
    background  the most common strong (saturated) color along the template
                border, if it covers a large part of the border. Every pixel
                of that color is masked.
    occluders   saturated connected regions that are tall (span most of the
                template height) but thin. That shape is a spear or a pole,
                not a head.
The mask is dilated a little so the black outline around a removed object
goes too.
"""

import cv2
import numpy as np

SAT_MIN = 100  # "strong color" saturation threshold (0..255)
VAL_MIN = 80
HUE_TOL = 8  # hue +- tolerance when matching the background color
BORDER_SHARE = 0.35  # background color must cover this much of the border
TALL = 0.8  # occluder spans at least this fraction of the height
THIN = 0.25  # and its average width is at most this fraction of the width
DILATE = 1  # pixels of dilation around removed regions


def _strong(hsv):
    return (hsv[:, :, 1] >= SAT_MIN) & (hsv[:, :, 2] >= VAL_MIN)


def _hue_near(h, hue):
    d = np.abs(h.astype(np.int32) - int(hue))
    d = np.minimum(d, 180 - d)  # hue wraps around at 180
    return d <= HUE_TOL


def background_mask(hsv):
    """Pixels matching the dominant strong color on the template border."""
    h = hsv[:, :, 0]
    strong = _strong(hsv)
    border = np.zeros(h.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_strong = border & strong
    if border_strong.sum() == 0:
        return np.zeros(h.shape, dtype=bool)
    hist = np.bincount(h[border_strong].ravel(), minlength=180)
    hue = int(np.argmax(hist))
    share = (border & strong & _hue_near(h, hue)).sum() / border.sum()
    if share < BORDER_SHARE:
        return np.zeros(h.shape, dtype=bool)
    return strong & _hue_near(h, hue)


def occluder_mask(hsv, background=None):
    """Tall, thin regions of strong color (spears, poles).

    Background pixels are removed first, so a pole drawn directly on a
    background of another strong color is not merged into it.
    """
    H, W = hsv.shape[:2]
    strong = _strong(hsv)
    if background is not None:
        strong = strong & ~background
    strong = strong.astype(np.uint8)
    out = np.zeros((H, W), dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=4)
    for i in range(1, n):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # Average width per row (area / height) instead of the bounding box
        # width, so a slanted spear still counts as thin.
        mean_width = stats[i, cv2.CC_STAT_AREA] / height
        if height >= TALL * H and mean_width <= THIN * W:
            out |= labels == i
    return out


def make_mask(template_bgr):
    """uint8 mask, 255 = keep pixel, 0 = ignore pixel."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    background = background_mask(hsv)
    occluders = occluder_mask(hsv, background)
    remove = background | occluders
    if DILATE > 0:
        kernel = np.ones((2 * DILATE + 1, 2 * DILATE + 1), dtype=np.uint8)
        grown = cv2.dilate(occluders.astype(np.uint8), kernel) > 0
        remove = remove | grown
    keep = ~remove
    return (keep * 255).astype(np.uint8)
