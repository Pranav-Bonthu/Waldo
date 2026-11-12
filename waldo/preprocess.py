"""Pre-processing applied identically to the template and the search image.

Every step is controlled by a config dict (see config.json), so the grid
search can turn steps on and off. The output of `apply_pipeline` is always a
float32 array with shape (H, W, C), where C is 1 (gray/edges) or 3 (color).

Order of operations:
    1. color space   gray, bgr or hsv
    2. hist eq       none, global (equalizeHist) or clahe
    3. smoothing     Gaussian blur
    4. edges         none, sobel (gradient magnitude) or canny (binary edges)

Edges are always computed on a gray image, so an edge pipeline ends with C = 1.
"""

import cv2
import numpy as np


def convert_color(image_bgr, color_space):
    """Return a uint8 image in the requested color space."""
    if color_space == "gray":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return gray[:, :, np.newaxis]
    if color_space == "bgr":
        return image_bgr.copy()
    if color_space == "hsv":
        # OpenCV hue runs 0..179, so red sits at BOTH ends of the range.
        # Correlating raw hue is crude for that reason, which is part of
        # what the experiment is meant to show.
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Unknown color_space: {color_space}")


def equalize(image, method, color_space, clahe_clip=2.0, clahe_tile=8):
    """Histogram equalization.

    For gray, equalize the single channel. For BGR, equalize each channel.
    For HSV, equalize only V (brightness) so hue is not distorted.
    """
    if method == "none":
        return image
    if method == "global":
        def eq(channel):
            return cv2.equalizeHist(channel)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))

        def eq(channel):
            return clahe.apply(channel)
    else:
        raise ValueError(f"Unknown hist_eq: {method}")

    out = image.copy()
    if color_space == "hsv":
        out[:, :, 2] = eq(np.ascontiguousarray(image[:, :, 2]))
        return out
    for c in range(image.shape[2]):
        out[:, :, c] = eq(np.ascontiguousarray(image[:, :, c]))
    return out


def smooth(image, ksize, sigma):
    """Gaussian blur. Reduces JPEG noise and fine texture before matching."""
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    if blurred.ndim == 2:
        blurred = blurred[:, :, np.newaxis]
    return blurred


def to_gray_uint8(image, color_space):
    if color_space == "gray":
        return image[:, :, 0]
    if color_space == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if color_space == "hsv":
        return image[:, :, 2]
    raise ValueError(f"Unknown color_space: {color_space}")


def edges(image, method, color_space, canny_low=50, canny_high=150):
    """Edge maps from the spatial filtering chapter.

    sobel: gradient magnitude sqrt(Gx^2 + Gy^2), scaled to 0..1
    canny: binary edge map (0 or 1)
    """
    gray = to_gray_uint8(image, color_space)
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        # Max possible 3x3 Sobel magnitude for uint8 input is about 1442.
        mag = mag / 1442.0
        return mag[:, :, np.newaxis].astype(np.float32)
    if method == "canny":
        e = cv2.Canny(gray, canny_low, canny_high)
        return (e[:, :, np.newaxis] / 255.0).astype(np.float32)
    raise ValueError(f"Unknown edges: {method}")


def to_float(image, color_space):
    """Scale uint8 channels to roughly 0..1 floats."""
    out = image.astype(np.float32)
    if color_space == "hsv":
        out[:, :, 0] = out[:, :, 0] / 180.0
        out[:, :, 1] = out[:, :, 1] / 255.0
        out[:, :, 2] = out[:, :, 2] / 255.0
        return out
    return out / 255.0


def apply_pipeline(image_bgr, cfg):
    """Run the configured pre-processing steps. Returns float32 (H, W, C)."""
    color_space = cfg.get("color_space", "gray")
    img = convert_color(image_bgr, color_space)

    img = equalize(
        img,
        cfg.get("hist_eq", "none"),
        color_space,
        cfg.get("clahe_clip", 2.0),
        cfg.get("clahe_tile", 8),
    )

    smoothing = cfg.get("smoothing", {})
    if smoothing.get("enabled", False):
        img = smooth(img, smoothing.get("ksize", 5), smoothing.get("sigma", 1.0))

    edge_method = cfg.get("edges", "none")
    if edge_method != "none":
        return edges(img, edge_method, color_space, cfg.get("canny_low", 50), cfg.get("canny_high", 150))

    return to_float(img, color_space)
