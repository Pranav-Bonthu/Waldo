"""Template matching from scratch.

All functions take:
    image     float32 array (H, W, C)
    template  float32 array (h, w, C)
and return a score map of shape (H - h + 1, W - w + 1). Entry [y, x] scores
the template placed with its top-left corner at (x, y).

Notation for one placement: T is the template, I is the image window under
it, N = h * w is the number of pixels per channel, and sums run over every
pixel and channel.

    ccorr  = sum(T * I)
             Plain cross-correlation. Biased toward bright regions, because a
             bright window makes the sum large even if it looks nothing like T.

    ssd    = sum((T - I)^2) = sum(T^2) - 2 sum(T * I) + sum(I^2)
             Sum of squared differences. 0 means a perfect match, so LOWER is
             better. (match.py negates it so that higher is always better.)

    ncc    = sum(T * I) / sqrt(sum(T^2) * sum(I^2))
             Normalized cross-correlation: the cosine of the angle between T
             and I viewed as long vectors. Brightness scaling no longer helps.
             Same as OpenCV TM_CCORR_NORMED.

    zncc   = sum((T - mean T)(I - mean I)) / sqrt(sum((T - mean T)^2) sum((I - mean I)^2))
             Zero-mean NCC: subtract each channel's mean first, so adding a
             constant brightness offset no longer helps either. Range -1..1.
             Same as OpenCV TM_CCOEFF_NORMED (means are per channel).

Two implementations are provided for each:
    *_loops   explicit Python double loop, easy to read, very slow
    fast path vectorized NumPy using sliding windows and integral images
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12
METRICS = ("ccorr", "ssd", "ncc", "zncc")


# ---------------------------------------------------------------------------
# Readable loop versions
# ---------------------------------------------------------------------------

def _score_window(window, template, metric):
    """Score one image window against the template using the formulas above."""
    if metric == "ccorr":
        return np.sum(window * template)

    if metric == "ssd":
        diff = window - template
        return np.sum(diff * diff)

    if metric == "ncc":
        num = np.sum(window * template)
        den = np.sqrt(np.sum(template * template) * np.sum(window * window))
        if den < EPS:
            return 0.0
        return num / den

    if metric == "zncc":
        # Subtract the mean of each channel separately (matches OpenCV).
        t0 = template - template.mean(axis=(0, 1), keepdims=True)
        w0 = window - window.mean(axis=(0, 1), keepdims=True)
        num = np.sum(t0 * w0)
        den = np.sqrt(np.sum(t0 * t0) * np.sum(w0 * w0))
        if den < EPS:
            return 0.0
        return num / den

    raise ValueError(f"Unknown metric: {metric}")


def match_loops(image, template, metric):
    """Slide the template over every position with a Python double loop."""
    H, W = image.shape[:2]
    h, w = template.shape[:2]
    out = np.zeros((H - h + 1, W - w + 1), dtype=np.float64)
    for y in range(H - h + 1):
        for x in range(W - w + 1):
            window = image[y:y + h, x:x + w]
            out[y, x] = _score_window(window, template, metric)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorized versions
# ---------------------------------------------------------------------------

def window_sums(a, h, w):
    """Sum of `a` over every h x w window, per channel, using an integral image.

    Integral image S: S[y, x] = sum of a[:y, :x]. Then any window sum is
        S[y+h, x+w] - S[y, x+w] - S[y+h, x] + S[y, x]
    which costs 4 lookups no matter how big the window is.

    a: (H, W, C)  ->  returns (H - h + 1, W - w + 1, C)
    """
    a = a.astype(np.float64)
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1, a.shape[2]), dtype=np.float64)
    S[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]


def cross_terms(image, template):
    """sum(T * I) for every placement, per channel. Returns (H', W', C).

    sliding_window_view gives a (H', W', C, h, w) view of all windows without
    copying, and einsum multiplies each window by the template and sums.
    """
    windows = sliding_window_view(image, template.shape[:2], axis=(0, 1))
    t = np.moveaxis(template, 2, 0)  # (C, h, w) to line up with the view
    return np.einsum("yxchw,chw->yxc", windows, t, optimize=True).astype(np.float64)


def match_numpy(image, template, metric):
    """Vectorized version of match_loops. Same output, much faster."""
    h, w = template.shape[:2]
    n = h * w
    t = template.astype(np.float64)

    TI = cross_terms(image, template)  # (H', W', C)

    if metric == "ccorr":
        return TI.sum(axis=2).astype(np.float32)

    I2 = window_sums(image * image, h, w)  # sum(I^2) per channel
    T2 = np.sum(t * t, axis=(0, 1))  # sum(T^2) per channel

    if metric == "ssd":
        ssd = T2.sum() - 2.0 * TI.sum(axis=2) + I2.sum(axis=2)
        return np.maximum(ssd, 0.0).astype(np.float32)

    if metric == "ncc":
        den = np.sqrt(T2.sum() * I2.sum(axis=2))
        out = np.where(den > EPS, TI.sum(axis=2) / np.maximum(den, EPS), 0.0)
        return out.astype(np.float32)

    if metric == "zncc":
        I1 = window_sums(image, h, w)  # sum(I) per channel
        T1 = np.sum(t, axis=(0, 1))  # sum(T) per channel
        # Expanding sum((T - Tbar)(I - Ibar)) gives sum(TI) - sum(T) sum(I) / N
        num = (TI - T1 * I1 / n).sum(axis=2)
        t_var = (T2 - T1 * T1 / n).sum()
        i_var = (I2 - I1 * I1 / n).sum(axis=2)
        den = np.sqrt(np.maximum(t_var * i_var, 0.0))
        out = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)
        return out.astype(np.float32)

    raise ValueError(f"Unknown metric: {metric}")
