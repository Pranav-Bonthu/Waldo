"""Cut Waldo's template (head + hat) out of every labeled scene.

Usage:
    python scripts/make_templates.py

Reads data/labels.json and writes data/templates/<n>.png for each entry.
These templates are what find_waldo.py searches for.
"""

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
ORIGINALS = ROOT / "data" / "original"
TEMPLATES = ROOT / "data" / "templates"
LABELS = ROOT / "data" / "labels.json"


def main():
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS.read_text())
    for name, entry in labels.items():
        if name.startswith("_"):
            continue
        x, y, w, h = entry["waldo"]
        scene = cv2.imread(str(ORIGINALS / name), cv2.IMREAD_COLOR)
        out = TEMPLATES / f"{Path(name).stem}.png"
        cv2.imwrite(str(out), scene[y:y + h, x:x + w])
        print(f"{name}: wrote {out.relative_to(ROOT)} ({w}x{h})")


if __name__ == "__main__":
    main()
