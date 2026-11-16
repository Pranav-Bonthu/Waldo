"""Tests run on crops of the real scenes so they finish in about a second."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waldo.data import ORIGINAL_DIR, TEMPLATE_DIR, load_labels, read_image  # noqa: E402
from waldo.evaluate import iou  # noqa: E402
from waldo.matcher import find_waldo  # noqa: E402
from waldo.template_mask import make_mask  # noqa: E402


def crop_around(scene, box, factor=10):
    """Square crop `factor` times Waldo's box size, centred on him. Returns crop and shifted box."""
    H, W = scene.shape[:2]
    x, y, w, h = box
    size = factor * max(w, h)
    x0 = min(max(0, x + w // 2 - size // 2), max(0, W - size))
    y0 = min(max(0, y + h // 2 - size // 2), max(0, H - size))
    return scene[y0:y0 + size, x0:x0 + size], [x - x0, y - y0, w, h]


@pytest.mark.parametrize("scene_name, template_name", [("9.jpg", "1.png"), ("1.jpg", "9.png")])
def test_finds_waldo_with_template_from_other_scene(scene_name, template_name):
    crop, truth = crop_around(read_image(ORIGINAL_DIR / scene_name), load_labels()[scene_name])
    best = find_waldo(crop, read_image(TEMPLATE_DIR / template_name))[0]
    assert iou(best["box"], truth) >= 0.5


def test_single_scale_exact_template():
    crop, truth = crop_around(read_image(ORIGINAL_DIR / "1.jpg"), load_labels()["1.jpg"])
    best = find_waldo(crop, read_image(TEMPLATE_DIR / "1.png"), scales=[1.0], use_mask=False)[0]
    assert iou(best["box"], truth) >= 0.5


def test_mask_removes_background_and_pole():
    """Yellow background and a tall thin red pole are masked, the rest kept."""
    t = np.full((50, 40, 3), (0, 220, 255), dtype=np.uint8)  # yellow (BGR)
    t[10:40, 12:34] = (180, 140, 120)  # a dull "face" block
    t[:, 4:9] = (0, 0, 230)  # red pole, full height, 5 px wide
    mask = make_mask(t)
    assert mask[25, 20] == 255  # face kept
    assert mask[2, 2] == 0  # background removed
    assert mask[25, 6] == 0  # pole removed
