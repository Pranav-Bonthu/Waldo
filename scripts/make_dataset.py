"""Build templates and zoomed crops from the labeled original images.

Usage:
    python scripts/make_dataset.py
    python scripts/make_dataset.py --zoom 5 --positives 5 --negatives 3 --seed 42

For every image in data/labels.json:
    data/templates/<n>.png      Waldo's head + hat, cut from the original
    data/zoomed/<n>_pos<i>.png  square crops (zoom x Waldo's box size) that
                                contain Waldo at a random position
    data/zoomed/<n>_neg<i>.png  same size crops that do NOT contain Waldo
    data/zoomed/manifest.json   where Waldo is in each crop (null for negatives)

A fixed seed makes the crops reproducible.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from waldo.data import (  # noqa: E402
    MANIFEST_PATH,
    ORIGINAL_DIR,
    TEMPLATE_DIR,
    ZOOMED_DIR,
    load_labels,
    read_image,
    save_json,
    template_path,
)
from waldo.evaluate import iou  # noqa: E402


def positive_crop(rng, image_shape, box, size):
    """Random square crop of `size` that fully contains `box`."""
    H, W = image_shape[:2]
    x, y, w, h = box
    x_min = max(0, x + w - size)
    x_max = min(x, W - size)
    y_min = max(0, y + h - size)
    y_max = min(y, H - size)
    cx = rng.randint(x_min, x_max)
    cy = rng.randint(y_min, y_max)
    return cx, cy


def negative_crop(rng, image_shape, box, size, max_tries=1000):
    """Random square crop of `size` that does not touch `box` at all."""
    H, W = image_shape[:2]
    for _ in range(max_tries):
        cx = rng.randint(0, W - size)
        cy = rng.randint(0, H - size)
        if iou([cx, cy, size, size], box) == 0.0:
            return cx, cy
    raise RuntimeError("Could not find a negative crop that avoids Waldo")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zoom", type=float, default=5.0, help="crop side = zoom x Waldo box size")
    parser.add_argument("--positives", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels = load_labels()
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    ZOOMED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "_format": "box = [x, y, w, h] of Waldo inside the crop, null if Waldo is absent",
        "zoom": args.zoom,
        "seed": args.seed,
        "crops": [],
    }

    for name, box in sorted(labels.items()):
        image = read_image(ORIGINAL_DIR / name)
        stem = Path(name).stem
        x, y, w, h = box

        tpl = image[y:y + h, x:x + w]
        cv2.imwrite(str(template_path(name)), tpl)

        size = int(round(args.zoom * max(w, h)))

        for i in range(args.positives):
            cx, cy = positive_crop(rng, image.shape, box, size)
            crop = image[cy:cy + size, cx:cx + size]
            file_name = f"{stem}_pos{i}.png"
            cv2.imwrite(str(ZOOMED_DIR / file_name), crop)
            manifest["crops"].append({
                "file": file_name,
                "source": name,
                "template": template_path(name).name,
                "origin": [cx, cy],
                "box": [x - cx, y - cy, w, h],
            })

        for i in range(args.negatives):
            cx, cy = negative_crop(rng, image.shape, box, size)
            crop = image[cy:cy + size, cx:cx + size]
            file_name = f"{stem}_neg{i}.png"
            cv2.imwrite(str(ZOOMED_DIR / file_name), crop)
            manifest["crops"].append({
                "file": file_name,
                "source": name,
                "template": template_path(name).name,
                "origin": [cx, cy],
                "box": None,
            })

        print(f"{name}: template {w}x{h}, crops {size}x{size}")

    save_json(manifest, MANIFEST_PATH)
    print(f"Wrote {len(manifest['crops'])} crops and {MANIFEST_PATH.relative_to(MANIFEST_PATH.parents[2])}")


if __name__ == "__main__":
    main()
