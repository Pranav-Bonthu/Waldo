"""Drawing the result and showing it in a window."""

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 255)
MAX_DISPLAY = 1000  # longest side of the scene panel, in pixels


def _scaled_box(box, scale):
    return [int(round(v * scale)) for v in box]


def draw_scene(scene_bgr, best, truth=None):
    """Scene resized to fit the screen, truth in green, best guess in red."""
    scale = min(3.0, MAX_DISPLAY / max(scene_bgr.shape[:2]))
    interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    out = cv2.resize(scene_bgr, None, fx=scale, fy=scale, interpolation=interp)
    if truth is not None:
        x, y, w, h = _scaled_box(truth, scale)
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 3)
    x, y, w, h = _scaled_box(best["box"], scale)
    cv2.rectangle(out, (x, y), (x + w, y + h), RED, 3)
    return out


def draw_side_panel(scene_bgr, best, truth=None, pad=0.6, panel_frac=0.4):
    """Scene with the best guess on the left, a zoomed crop of that guess on the right."""
    left = draw_scene(scene_bgr, best, truth)
    left_h = left.shape[0]

    # Crop around the best box with some context, clamped to the image.
    H, W = scene_bgr.shape[:2]
    x, y, w, h = best["box"]
    px, py = int(round(w * pad)), int(round(h * pad))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    crop = scene_bgr[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), RED, 1)

    # Enlarge the crop to fill the panel, keeping its aspect ratio.
    header = 40
    panel_w = max(120, int(round(left.shape[1] * panel_frac)))
    zoom_scale = min(panel_w / crop.shape[1], (left_h - header) / crop.shape[0])
    zoom = cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_CUBIC)

    panel = np.full((left_h, panel_w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "Best guess", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"score {best['score']:.2f}", (panel_w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
    ox = (panel_w - zoom.shape[1]) // 2
    oy = header + (left_h - header - zoom.shape[0]) // 2
    panel[oy:oy + zoom.shape[0], ox:ox + zoom.shape[1]] = zoom

    separator = np.full((left_h, 4, 3), 90, dtype=np.uint8)
    return np.hstack([left, separator, panel])


def show(window_name, image):
    """Pop-up window. Press any key to close."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
