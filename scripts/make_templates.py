"""Cut Waldo's template (head + hat) out of every labeled scene.

Usage:
    python scripts/make_templates.py

Reads data/labels.json and writes data/templates/<n>.png for each entry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from waldo.data import ORIGINAL_DIR, TEMPLATE_DIR, load_labels, read_image, template_path  # noqa: E402


def main():
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    for name, (x, y, w, h) in load_labels().items():
        scene = read_image(ORIGINAL_DIR / name)
        out = template_path(name)
        cv2.imwrite(str(out), scene[y:y + h, x:x + w])
        print(f"{name}: wrote {out.relative_to(TEMPLATE_DIR.parent.parent)} ({w}x{h})")


if __name__ == "__main__":
    main()
