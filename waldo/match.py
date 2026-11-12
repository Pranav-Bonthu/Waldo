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


def raw_score_map(image, template, metric, backend="opencv", mask=None):
    """Score map in the metric's natural units (SSD: lower is better).

    mask (optional, OpenCV backend only): uint8 array the size of the
    template, 255 = use pixel, 0 = ignore pixel.
    """
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError("Template is larger than the image")

    if backend == "opencv":
        img = np.ascontiguousarray(image, dtype=np.float32)
        tpl = np.ascontiguousarray(template, dtype=np.float32)
        # OpenCV expects 2D arrays for single channel images.
        if img.shape[2] == 1:
            img = img[:, :, 0]
            tpl = tpl[:, :, 0]
        if mask is None:
            return cv2.matchTemplate(img, tpl, OPENCV_METHODS[metric])
        scores = cv2.matchTemplate(img, tpl, OPENCV_METHODS[metric], mask=mask)
        # Masked matching can divide by zero on flat windows. Replace any
        # NaN/inf with the worst finite score so they never win.
        finite = np.isfinite(scores)
        worst = scores[finite].max() if metric == "ssd" else scores[finite].min()
        return np.where(finite, scores, worst).astype(np.float32)

    if mask is not None:
        raise ValueError("Template masks are only supported with the OpenCV backend")

    if backend == "numpy":
        return correlation.match_numpy(image, template, metric)

    if backend == "loops":
        return correlation.match_loops(image, template, metric)

    raise ValueError(f"Unknown backend: {backend}")


def score_map(image, template, metric, backend="opencv", mask=None):
    """Score map where higher is always better (SSD is negated)."""
    scores = raw_score_map(image, template, metric, backend, mask)
    if metric == "ssd":
        return -scores
    return scores
