"""Find Waldo in a scene. This is the one file to run.

    python find_waldo.py                      # searches SCENE below
    python find_waldo.py path/to/any.jpg      # searches that image instead

Every template in data/templates (Waldo's head and hat, cut from labeled
scenes) is slid over the scene at many sizes. The single best-scoring spot
across all templates is the answer. A window shows the scene with that box
in red (green = the labeled answer, if the scene has one) and a zoomed crop
on the right. Press any key to close it.
"""

import json
import sys
import time
from pathlib import Path

import cv2

from waldo.matcher import find_waldo, iou
from waldo.visualize import draw_side_panel, show

SCENE = "data/original/9.jpg"  # change this, or pass a path on the command line
SHOW_WINDOW = True

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "data" / "templates"
ORIGINALS = ROOT / "data" / "original"
LABELS = ROOT / "data" / "labels.json"


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def labeled_box(scene_path):
    """Waldo's labeled box if this scene is one of data/original, else None."""
    path = Path(scene_path).resolve()
    if path.parent != ORIGINALS or not LABELS.exists():
        return None
    entry = json.loads(LABELS.read_text()).get(path.name)
    return None if entry is None else entry["waldo"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scene_path = args[0] if args else SCENE
    scene = read_image(scene_path)
    templates = sorted(TEMPLATES.glob("*.png"))
    if not templates:
        raise SystemExit("No templates in data/templates. Run scripts/label_waldo.py then scripts/make_templates.py.")

    start = time.perf_counter()
    best = None
    for tpl_path in templates:
        top = find_waldo(scene, read_image(tpl_path))
        if top and (best is None or top[0]["score"] > best["score"]):
            best = dict(top[0], template=tpl_path.name)
    seconds = time.perf_counter() - start

    print(f"Scene: {scene_path}   templates: {len(templates)}   time: {seconds:.1f} s")
    print(f"  Waldo: box={best['box']}  score={best['score']:.2f}  size={best['scale']:.2f}x  template={best['template']}")

    truth = labeled_box(scene_path)
    if truth is not None:
        overlap = iou(best["box"], truth)
        print(f"  Label {truth}: {'CORRECT' if overlap >= 0.5 else 'WRONG'} (overlap {overlap:.2f})")

    if SHOW_WINDOW and "--no-show" not in sys.argv:
        show("Where's Waldo (any key to close)", draw_side_panel(scene, best, truth))


if __name__ == "__main__":
    main()
