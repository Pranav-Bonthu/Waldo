"""Glue: pre-process, match, pick peaks, optionally color-filter."""

import time

from waldo import color_mask
from waldo.match import score_map
from waldo.peaks import top_k_peaks
from waldo.preprocess import apply_pipeline


def detect(image_bgr, template_bgr, cfg):
    """Find Waldo candidates in image_bgr.

    Returns a dict:
        candidates   ranked list of {box, score}, length <= cfg["top_k"]
        best_score   score of the top candidate (before color re-ranking)
        found        False if best_score is below cfg["not_found_threshold"]
        seconds      runtime
    """
    start = time.perf_counter()

    img = apply_pipeline(image_bgr, cfg)
    tpl = apply_pipeline(template_bgr, cfg)
    scores = score_map(img, tpl, cfg.get("metric", "zncc"), cfg.get("backend", "opencv"))

    top_k = cfg.get("top_k", 5)
    color_cfg = cfg.get("color_filter", {})
    use_color = color_cfg.get("enabled", False)

    # With the color filter we look at extra peaks, since some get demoted.
    n_peaks = top_k * 4 if use_color else top_k
    candidates = top_k_peaks(scores, tpl.shape[:2], n_peaks, cfg.get("nms", True))

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
        "scores": scores,
        "seconds": time.perf_counter() - start,
    }
