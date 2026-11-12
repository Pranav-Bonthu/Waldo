"""Glue: pre-process, match, pick peaks, optionally color-filter.

Multi-scale matching
--------------------
Template matching compares pixel by pixel at ONE size. If Waldo is drawn
25% smaller in the search image than in the template, the patterns never
line up and his real location can score lower than random clutter.

The fix: resize the template over a range of scales (e.g. 0.6x to 1.4x),
run the match at every scale, and keep the best results across all of them.
This is the "template pyramid" version of an image pyramid: we shrink and
grow the small template instead of the big image, which is cheaper.

Scores must be comparable across scales:
    ccorr, ssd  grow with the number of template pixels, so divide by it
                (this turns them into a per-pixel average)
    all         even NCC is not fair across sizes: a small template matches
                random clutter more easily. By default every scale's map is
                z-scored, see normalize_scores().
"""

import time

import cv2
import numpy as np

from waldo import color_mask
from waldo.evaluate import iou
from waldo.match import score_map
from waldo.peaks import top_k_peaks
from waldo.preprocess import apply_pipeline
from waldo.template_mask import make_mask


def template_scales(cfg):
    """List of scale factors to try. [1.0] when multi-scale is off."""
    ms = cfg.get("multiscale", {})
    if not ms.get("enabled", False):
        return [1.0]
    return list(np.linspace(ms.get("min", 0.6), ms.get("max", 1.4), ms.get("steps", 17)))


def resize_template(template_bgr, scale):
    if scale == 1.0:
        return template_bgr
    h, w = template_bgr.shape[:2]
    new_w = max(4, int(round(w * scale)))
    new_h = max(4, int(round(h * scale)))
    # INTER_AREA averages pixels when shrinking (avoids aliasing),
    # INTER_CUBIC interpolates smoothly when enlarging.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(template_bgr, (new_w, new_h), interpolation=interp)


def normalize_scores(scores, tpl, metric, method):
    """Make score maps from different template sizes comparable.

    none    only divide ccorr/ssd by the pixel count (ncc/zncc unchanged)
    zscore  (score - mean) / std of that scale's score map. A peak's value
            becomes "how many standard deviations above typical". Small
            templates match random clutter more easily, so their maps are
            noisier and their raw peaks are inflated. Z-scoring cancels that.
    """
    if metric in ("ccorr", "ssd"):
        scores = scores / float(tpl.size)
    if method == "none":
        return scores
    if method == "zscore":
        std = float(scores.std())
        if std < 1e-12:
            return scores * 0.0
        return (scores - float(scores.mean())) / std
    raise ValueError(f"Unknown multiscale normalize: {method}")


def merge_candidates(candidates, iou_limit=0.3):
    """Non-max suppression across scales.

    Boxes from different scales have different sizes, so instead of blanking
    a window in one score map, drop any box that overlaps a better-scoring
    box by IoU > iou_limit.
    """
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    for c in ordered:
        overlaps = False
        for k in kept:
            if iou(c["box"], k["box"]) > iou_limit:
                overlaps = True
                break
        if not overlaps:
            kept.append(c)
    return kept


def detect(image_bgr, template_bgr, cfg):
    """Find Waldo candidates in image_bgr.

    Returns a dict:
        candidates   ranked list of {box, score, scale}, length <= cfg["top_k"]
        best_score   score of the top candidate (before color re-ranking)
        found        False if best_score is below cfg["not_found_threshold"]
        scores       score map of the scale that produced the best match
        seconds      runtime
    """
    start = time.perf_counter()

    metric = cfg.get("metric", "zncc")
    top_k = cfg.get("top_k", 5)
    color_cfg = cfg.get("color_filter", {})
    use_color = color_cfg.get("enabled", False)
    scales = template_scales(cfg)

    # With the color filter we look at extra peaks, since some get demoted.
    n_peaks = top_k * 4 if use_color else top_k

    img = apply_pipeline(image_bgr, cfg)
    H, W = img.shape[:2]

    # Optional mask: ignore template pixels that are not Waldo.
    base_mask = None
    if cfg.get("template_mask", {}).get("enabled", False):
        base_mask = make_mask(template_bgr)

    candidates = []
    best_map = None
    best_map_score = float("-inf")
    for scale in scales:
        tpl_bgr = resize_template(template_bgr, scale)
        if tpl_bgr.shape[0] > H or tpl_bgr.shape[1] > W:
            continue
        tpl = apply_pipeline(tpl_bgr, cfg)
        mask = None
        if base_mask is not None:
            size = (tpl_bgr.shape[1], tpl_bgr.shape[0])
            mask = cv2.resize(base_mask, size, interpolation=cv2.INTER_NEAREST)
        scores = score_map(img, tpl, metric, cfg.get("backend", "opencv"), mask)
        if len(scales) > 1:
            scores = normalize_scores(scores, tpl, metric, cfg["multiscale"].get("normalize", "zscore"))
        peaks = top_k_peaks(scores, tpl.shape[:2], n_peaks, cfg.get("nms", True))
        for p in peaks:
            p["scale"] = float(scale)
        candidates.extend(peaks)
        if peaks and peaks[0]["score"] > best_map_score:
            best_map_score = peaks[0]["score"]
            best_map = scores

    if len(scales) > 1:
        candidates = merge_candidates(candidates)[:n_peaks]

    best_score = candidates[0]["score"] if candidates else float("-inf")

    if use_color:
        ranges = color_mask.learn_ranges(template_bgr)
        candidates = color_mask.rerank(image_bgr, candidates, ranges, color_cfg.get("min_ratio", 0.5))

    candidates = candidates[:top_k]

    threshold = cfg.get("not_found_threshold")
    found = True
    if threshold is not None:
        found = best_score >= threshold

    return {
        "candidates": candidates,
        "best_score": best_score,
        "found": found,
        "scores": best_map,
        "seconds": time.perf_counter() - start,
    }
