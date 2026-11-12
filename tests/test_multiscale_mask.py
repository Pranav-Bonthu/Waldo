"""Tests for multi-scale matching and the automatic template mask.

Run with:  python -m pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from waldo.data import load_config  # noqa: E402
from waldo.detector import detect  # noqa: E402
from waldo.evaluate import iou  # noqa: E402
from waldo.template_mask import make_mask  # noqa: E402


def textured_image(seed, shape=(200, 260)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (shape[0] // 4, shape[1] // 4, 3), dtype=np.uint8)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def test_multiscale_finds_smaller_copy():
    """Template is 1.25x bigger than the object in the image."""
    image = textured_image(0)
    x, y, w, h = 120, 60, 32, 40
    obj = image[y:y + h, x:x + w].copy()
    template = cv2.resize(obj, (40, 50), interpolation=cv2.INTER_NEAREST)
    cfg = load_config()
    cfg["color_space"] = "bgr"
    cfg["multiscale"]["enabled"] = True
    best = detect(image, template, cfg)["candidates"][0]
    assert iou(best["box"], [x, y, w, h]) >= 0.5


def test_mask_removes_background_and_pole():
    """Yellow background and a tall thin red pole are masked, the rest kept."""
    t = np.full((50, 40, 3), (0, 220, 255), dtype=np.uint8)  # yellow (BGR)
    t[10:40, 12:34] = (180, 140, 120)  # a dull "face" block
    t[:, 4:9] = (0, 0, 230)  # red pole, full height, 5 px wide
    mask = make_mask(t)
    assert mask[25, 20] == 255  # face kept
    assert mask[2, 37] == 0  # yellow corner removed
    assert mask[25, 6] == 0  # pole removed


def test_mask_keeps_plain_template():
    t = np.full((30, 20, 3), 200, dtype=np.uint8)
    assert (make_mask(t) == 255).all()
