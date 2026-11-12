"""Checks that the from-scratch matching is correct.

Run with:  python -m pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from waldo.correlation import METRICS, match_loops, match_numpy  # noqa: E402
from waldo.evaluate import iou  # noqa: E402
from waldo.match import raw_score_map, score_map  # noqa: E402
from waldo.peaks import top_k_peaks  # noqa: E402


def random_pair(channels, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random((40, 50, channels)).astype(np.float32)
    template = rng.random((7, 9, channels)).astype(np.float32)
    return image, template


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("channels", [1, 3])
def test_loops_equal_numpy(metric, channels):
    image, template = random_pair(channels)
    a = match_loops(image, template, metric)
    b = match_numpy(image, template, metric)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("channels", [1, 3])
def test_numpy_matches_opencv(metric, channels):
    image, template = random_pair(channels, seed=1)
    ours = raw_score_map(image, template, metric, backend="numpy")
    cv = raw_score_map(image, template, metric, backend="opencv")
    np.testing.assert_allclose(ours, cv, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("metric", ["ssd", "ncc", "zncc"])
def test_pasted_template_is_found(metric):
    """Cut a patch out of a random image and check we find it exactly."""
    rng = np.random.default_rng(2)
    image = rng.random((60, 80, 3)).astype(np.float32)
    x, y, w, h = 33, 17, 10, 12
    template = image[y:y + h, x:x + w].copy()
    for backend in ["opencv", "numpy"]:
        scores = score_map(image, template, metric, backend)
        best = top_k_peaks(scores, (h, w), k=1)[0]
        assert best["box"] == [x, y, w, h]


def test_nms_spreads_peaks():
    scores = np.zeros((50, 50), dtype=np.float32)
    scores[10, 10] = 1.0
    scores[11, 11] = 0.9  # right next to the best peak, should be suppressed
    scores[40, 40] = 0.5
    peaks = top_k_peaks(scores, (5, 5), k=2)
    assert peaks[0]["box"][:2] == [10, 10]
    assert peaks[1]["box"][:2] == [40, 40]


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 10, 10]) - 50 / 150) < 1e-9
