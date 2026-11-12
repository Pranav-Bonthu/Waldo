"""Where's Waldo finder using classical image processing.

Modules:
    data         loading labels, templates and zoomed crops
    preprocess   color spaces, smoothing, histogram equalization, edges
    correlation  from-scratch template matching (loops and vectorized NumPy)
    match        one entry point that runs OpenCV or NumPy matching
    peaks        picking the top-k locations with non-max suppression
    color_mask   red/white color filter learned from the template
    evaluate     IoU, top-5 accuracy and negative rejection
"""
