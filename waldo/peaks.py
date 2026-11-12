"""Turn a score map into a ranked list of candidate boxes."""

import numpy as np


def top_k_peaks(scores, template_hw, k=5, nms=True):
    """Return up to k candidates as dicts {box: [x, y, w, h], score}.

    Non-max suppression (NMS): after taking the best peak, blank out every
    position within one template size of it. Without this, the "top 5" would
    usually be the same spot shifted by a pixel or two.
    """
    h, w = template_hw
    s = scores.astype(np.float64).copy()
    candidates = []
    for _ in range(k):
        if not np.isfinite(s).any():
            break
        idx = int(np.nanargmax(np.where(np.isfinite(s), s, -np.inf)))
        y, x = np.unravel_index(idx, s.shape)
        candidates.append({"box": [int(x), int(y), int(w), int(h)], "score": float(scores[y, x])})
        if nms:
            y0 = max(0, y - h + 1)
            y1 = y + h
            x0 = max(0, x - w + 1)
            x1 = x + w
            s[y0:y1, x0:x1] = -np.inf
        else:
            s[y, x] = -np.inf
    return candidates
