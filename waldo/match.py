"""Single entry point for template matching.

score_map() always returns a map where HIGHER means a better match, so the
rest of the code never needs to know which metric was used.
"""

import cv2
import numpy as np

from waldo import correlation

OPENCV_METHODS = {
    "ccorr": cv2.TM_CCORR,
    "ssd": cv2.TM_SQDIFF,
    "ncc": cv2.TM_CCORR_NORMED,
    "zncc": cv2.TM_CCOEFF_NORMED,
}


def raw_score_map(image, template, metric, backend="opencv"):
    """Score map in the metric's natural units (SSD: lower is better)."""
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError("Template is larger than the image")

    if backend == "opencv":
        img = np.ascontiguousarray(image, dtype=np.float32)
        tpl = np.ascontiguousarray(template, dtype=np.float32)
        # OpenCV expects 2D arrays for single channel images.
        if img.shape[2] == 1:
            img = img[:, :, 0]
            tpl = tpl[:, :, 0]
        return cv2.matchTemplate(img, tpl, OPENCV_METHODS[metric])

    if backend == "numpy":
        return correlation.match_numpy(image, template, metric)

    if backend == "loops":
        return correlation.match_loops(image, template, metric)

    raise ValueError(f"Unknown backend: {backend}")


def score_map(image, template, metric, backend="opencv"):
    """Score map where higher is always better (SSD is negated)."""
    scores = raw_score_map(image, template, metric, backend)
    if metric == "ssd":
        return -scores
    return scores
