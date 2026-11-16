"""Multi-scale template matching: the core of the finder.

    1. Pre-process the scene and the template the same way (preprocess.py).
    2. Build a mask of the template so background and poles do not count
       (template_mask.py).
    3. Resize the template to several sizes, 0.6x to 1.4x. Waldo is not drawn
       at the same size in every scene, and template matching only works when
       the sizes line up.
    4. At every size, slide the template over the whole scene. Each position
       gets a score: zero-mean normalized cross-correlation (ZNCC), which is
       -1..1 and ignores brightness differences. OpenCV does the sliding
       (cv2.matchTemplate with TM_CCOEFF_NORMED).
    5. Z-score each size's map ((score - mean) / std) so sizes can be compared
       fairly. A small template matches random clutter more easily, so its
       raw peaks are inflated.
    6. Take the best peaks at every size, drop any box that overlaps a
       better-scoring one, and return the top few. Candidate #1 is Waldo.
"""

import time

import cv2
import numpy as np

from waldo.evaluate import iou
from waldo.preprocess import preprocess
from waldo.template_mask import make_mask

SCALES = np.linspace(0.6, 1.4, 17)


def resize_template(template, scale, mask=None):
    """Resize the template (and its mask) by a factor. INTER_AREA when shrinking
    averages pixels so nothing aliases; INTER_CUBIC when growing stays smooth."""
    h, w = template.shape[:2]
    size = (max(4, int(round(w * scale))), max(4, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    small = cv2.resize(template, size, interpolation=interp)
    small_mask = None if mask is None else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return small, small_mask


def score_map(scene, template, mask=None):
    """ZNCC score of the template at every position. Higher = better match."""
    if mask is None:
        scores = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    else:
        scores = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        # Masked matching can divide by zero on flat windows. Give those
        # positions the worst finite score so they never win.
        finite = np.isfinite(scores)
        if not finite.all():
            scores = np.where(finite, scores, scores[finite].min()).astype(np.float32)
    return scores


def zscore(scores):
    std = float(scores.std())
    if std < 1e-12:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


def top_peaks(scores, template_hw, k):
    """The k highest scores as boxes, each at least one template apart.

    After taking the best position, everything within one template size of
    it is blanked out. Without that, the top 5 would be the same spot shifted
    by a pixel or two.
    """
    h, w = template_hw
    s = scores.copy()
    peaks = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[y, x] == -np.inf:
            break
        peaks.append({"box": [int(x), int(y), int(w), int(h)], "score": float(s[y, x])})
        s[max(0, y - h + 1):y + h, max(0, x - w + 1):x + w] = -np.inf
    return peaks


def merge(candidates, iou_limit=0.3):
    """Keep the best-scoring boxes, dropping any that overlaps one already kept."""
    kept = []
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if all(iou(c["box"], k["box"]) <= iou_limit for k in kept):
            kept.append(c)
    return kept


def find_waldo(scene_bgr, template_bgr, scales=SCALES, top_k=5, use_mask=True):
    """Return the top_k candidate boxes, best first, each {box, score, scale}."""
    scene = preprocess(scene_bgr)
    mask = make_mask(template_bgr) if use_mask else None

    candidates = []
    for scale in scales:
        tpl_bgr, tpl_mask = resize_template(template_bgr, scale, mask)
        if tpl_bgr.shape[0] > scene.shape[0] or tpl_bgr.shape[1] > scene.shape[1]:
            continue
        scores = score_map(scene, preprocess(tpl_bgr), tpl_mask)
        if len(scales) > 1:
            scores = zscore(scores)
        for p in top_peaks(scores, tpl_bgr.shape[:2], top_k):
            p["scale"] = float(scale)
            candidates.append(p)

    return merge(candidates)[:top_k]


def find_waldo_timed(scene_bgr, template_bgr, **kwargs):
    start = time.perf_counter()
    candidates = find_waldo(scene_bgr, template_bgr, **kwargs)
    return candidates, time.perf_counter() - start
