"""Random crops around (or away from) Waldo's box."""


def iou_zero(a, b):
    """True if boxes a and b do not overlap at all."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    return overlap_x <= 0 or overlap_y <= 0


def positive_crop(rng, image_shape, box, size):
    """Top-left corner of a random square crop of `size` that fully contains `box`.

    If `size` is bigger than the image in a direction, the crop is clipped to
    the image, so the returned crop may be smaller than size x size.
    """
    H, W = image_shape[:2]
    x, y, w, h = box
    size_x = min(size, W)
    size_y = min(size, H)
    x_min = max(0, x + w - size_x)
    x_max = min(x, W - size_x)
    y_min = max(0, y + h - size_y)
    y_max = min(y, H - size_y)
    cx = rng.randint(x_min, x_max)
    cy = rng.randint(y_min, y_max)
    return cx, cy, size_x, size_y


def negative_crop(rng, image_shape, box, size, max_tries=2000):
    """Top-left corner of a random square crop that does not touch `box`.

    Returns None if no such crop fits in the image (common for big crops).
    """
    H, W = image_shape[:2]
    if size > W or size > H:
        return None
    for _ in range(max_tries):
        cx = rng.randint(0, W - size)
        cy = rng.randint(0, H - size)
        if iou_zero([cx, cy, size, size], box):
            return cx, cy
    return None
