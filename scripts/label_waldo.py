"""Click-and-drag tool to label (or check) Waldo's box in an original image.

Usage:
    python scripts/label_waldo.py 1.jpg

A window opens with the image (scaled to fit your screen). If the image is
already labeled, the current box is drawn in green.
    - Drag a box around Waldo's head and hat, then press ENTER or SPACE to save.
    - Press C (or ESC) to cancel and keep the existing label.
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
ORIGINALS = ROOT / "data" / "original"
LABELS = ROOT / "data" / "labels.json"
MAX_WINDOW = 1200  # longest side of the displayed image, in pixels


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    image = cv2.imread(str(ORIGINALS / name), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {ORIGINALS / name}")
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}

    scale = min(1.0, MAX_WINDOW / max(image.shape[:2]))
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    existing = labels.get(name, {}).get("waldo")
    if existing is not None:
        x, y, w, h = [int(round(v * scale)) for v in existing]
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Existing label for {name}: {existing}")

    window = f"Label Waldo in {name} (ENTER to save, C to cancel)"
    roi = cv2.selectROI(window, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("No box drawn, label unchanged.")
        return

    box = [int(round(v / scale)) for v in roi]
    labels[name] = {"waldo": box}
    LABELS.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Saved {name}: {box}")


if __name__ == "__main__":
    main()
