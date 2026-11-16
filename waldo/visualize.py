"""Drawing boxes and showing pop-up windows."""

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 255)


MAX_DISPLAY = 1000  # longest side of the pop-up, in pixels


def draw_result(image_bgr, candidates, truth=None, scale=None, max_guesses=1):
    """Resize to fit the screen and draw truth (green) and ranked guesses (red).

    Only the top `max_guesses` guesses are drawn (default: just the #1 guess).

    Small crops are enlarged (up to 3x) so Waldo is visible. Full scenes are
    shrunk so the window fits on screen.
    """
    if scale is None:
        scale = min(3.0, MAX_DISPLAY / max(image_bgr.shape[:2]))
    interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    out = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)

    if truth is not None:
        x, y, w, h = [int(round(v * scale)) for v in truth]
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 3)

    for rank, c in enumerate(candidates[:max_guesses], start=1):
        x, y, w, h = [int(round(v * scale)) for v in c["box"]]
        thickness = 3 if rank == 1 else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), RED, thickness)
        label = f"{rank}: {c['score']:.2f}"
        cv2.putText(out, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

    return out


def show(window_name, image):
    """Pop-up window. Press any key to close.

    Uses OpenCV's window if available. Some OpenCV installs (the "headless"
    builds) have no window support, so fall back to a Matplotlib window.
    """
    try:
        cv2.imshow(window_name, image)
    except cv2.error:
        show_matplotlib(window_name, image)
        return
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def show_matplotlib(window_name, image):
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(window_name)
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def draw_side_panel(image_bgr, best, truth=None, pad=0.6, panel_frac=0.4):
    """Scene with the #1 guess on the left, a zoomed crop of that guess on the right.

    best    the top candidate dict {box, score, ...}
    pad     how much context to show around the box (0.6 = 60% of the box
            size on every side)
    panel_frac  width of the right panel as a fraction of the left panel
    """
    left = draw_result(image_bgr, [best], truth, max_guesses=1)
    left_h = left.shape[0]

    # Crop around the best box with some context, clamped to the image.
    H, W = image_bgr.shape[:2]
    x, y, w, h = best["box"]
    px, py = int(round(w * pad)), int(round(h * pad))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    crop = image_bgr[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), RED, 1)

    # Enlarge the crop to fill the panel height, keeping its aspect ratio.
    header = 40
    panel_w = max(120, int(round(left.shape[1] * panel_frac)))
    scale = min(panel_w / crop.shape[1], (left_h - header) / crop.shape[0])
    zoom = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    panel = np.full((left_h, panel_w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "Best guess", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"score {best['score']:.2f}", (panel_w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
    ox = (panel_w - zoom.shape[1]) // 2
    oy = header + (left_h - header - zoom.shape[0]) // 2
    panel[oy:oy + zoom.shape[0], ox:ox + zoom.shape[1]] = zoom

    separator = np.full((left_h, 4, 3), 90, dtype=np.uint8)
    return np.hstack([left, separator, panel])
