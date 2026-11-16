"""Where's Waldo finder using template matching.

Modules:
    data           paths, image reading, Waldo labels
    preprocess     CLAHE + blur applied to both the scene and the template
    template_mask  ignore template pixels that are not Waldo
    matcher        multi-scale template matching, the core of the finder
    evaluate       IoU, used to merge boxes and to check against the truth
    visualize      draw the result and show the pop-up window
"""
