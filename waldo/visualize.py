"""Drawing boxes and showing pop-up windows."""

import cv2

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
