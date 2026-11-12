"""Click-and-drag tool to label (or check) Waldo's box in an original image.

Usage:
    python scripts/label_waldo.py 1.jpg

A window opens with the image (scaled to fit your screen). If the image is
already labeled, the current box is drawn in green.
    - Drag a box around Waldo's head and hat, then press ENTER or SPACE to save.
    - Press C (or ESC) to cancel and keep the existing label.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from waldo.data import LABELS_PATH, ORIGINAL_DIR, load_json, read_image, save_json  # noqa: E402

MAX_WINDOW = 1200  # longest side of the displayed image, in pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="file name inside data/original, e.g. 1.jpg")
    args = parser.parse_args()

    image = read_image(ORIGINAL_DIR / args.image)
    labels = load_json(LABELS_PATH)

    scale = min(1.0, MAX_WINDOW / max(image.shape[:2]))
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    existing = labels.get(args.image, {}).get("waldo")
    if existing is not None:
        x, y, w, h = [int(round(v * scale)) for v in existing]
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Existing label for {args.image}: {existing}")

    window = f"Label Waldo in {args.image} (ENTER to save, C to cancel)"
    roi = cv2.selectROI(window, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("No box drawn, label unchanged.")
        return

    box = [int(round(v / scale)) for v in roi]
    labels[args.image] = {"waldo": box}
    save_json(labels, LABELS_PATH)
    print(f"Saved {args.image}: {box}")


if __name__ == "__main__":
    main()
