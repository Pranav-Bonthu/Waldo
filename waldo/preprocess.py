"""Prepare an image for matching. The scene and the template get the same treatment.

    1. keep BGR color      Waldo's red and white stripes are a big part of his look
    2. CLAHE per channel   local histogram equalization, evens out contrast so a
                           darker or lighter print still matches
    3. Gaussian blur       softens JPEG noise and fine texture so two slightly
                           different drawings of Waldo still line up
    4. float32 in 0..1     what cv2.matchTemplate wants
"""

import cv2
import numpy as np


def preprocess(image_bgr, clahe_clip=2.0, clahe_tile=8, blur_ksize=5, blur_sigma=1.0):
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    out = cv2.merge([clahe.apply(c) for c in cv2.split(image_bgr)])
    out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), blur_sigma)
    return out.astype(np.float32) / 255.0
