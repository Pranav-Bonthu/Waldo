"""Find Waldo in one zoomed crop and show the result in a pop-up window.

Usage:
    python scripts/find_waldo.py data/zoomed/1_pos0.png
    python scripts/find_waldo.py data/zoomed/9_pos2.png --metric ncc --color-space bgr
    python scripts/find_waldo.py data/zoomed/1_neg0.png --threshold 0.6

Settings come from config.json. Flags override individual settings.
Green box = true Waldo location (if known), red boxes = ranked guesses.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waldo.data import MANIFEST_PATH, TEMPLATE_DIR, load_config, load_json, read_image  # noqa: E402
from waldo.detector import detect  # noqa: E402
from waldo.evaluate import hit_rank  # noqa: E402
from waldo.visualize import draw_result, show  # noqa: E402


def find_manifest_entry(image_path):
    if not MANIFEST_PATH.exists():
        return None
    manifest = load_json(MANIFEST_PATH)
    for entry in manifest["crops"]:
        if entry["file"] == Path(image_path).name:
            return entry
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to a zoomed crop")
    parser.add_argument("--template", help="template image (default: from manifest)")
    parser.add_argument("--config", default=None, help="config file (default: config.json)")
    parser.add_argument("--metric", choices=["ccorr", "ssd", "ncc", "zncc"])
    parser.add_argument("--color-space", choices=["gray", "bgr", "hsv"])
    parser.add_argument("--hist-eq", choices=["none", "global", "clahe"])
    parser.add_argument("--edges", choices=["none", "sobel", "canny"])
    parser.add_argument("--blur", action="store_true", help="turn on Gaussian smoothing")
    parser.add_argument("--color-filter", action="store_true", help="turn on red/white re-ranking")
    parser.add_argument("--backend", choices=["opencv", "numpy", "loops"])
    parser.add_argument("--threshold", type=float, help="not-found score threshold")
    parser.add_argument("--no-show", action="store_true", help="print only, no pop-up window")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    if args.metric:
        cfg["metric"] = args.metric
    if args.color_space:
        cfg["color_space"] = args.color_space
    if args.hist_eq:
        cfg["hist_eq"] = args.hist_eq
    if args.edges:
        cfg["edges"] = args.edges
    if args.blur:
        cfg["smoothing"]["enabled"] = True
    if args.color_filter:
        cfg["color_filter"]["enabled"] = True
    if args.backend:
        cfg["backend"] = args.backend
    if args.threshold is not None:
        cfg["not_found_threshold"] = args.threshold

    entry = find_manifest_entry(args.image)
    if args.template:
        template_file = Path(args.template)
    elif entry is not None:
        template_file = TEMPLATE_DIR / entry["template"]
    else:
        parser.error("No --template given and the image is not in the manifest")

    image = read_image(args.image)
    template = read_image(template_file)
    result = detect(image, template, cfg)

    truth = entry["box"] if entry is not None else None

    print(f"Image: {args.image}   template: {template_file.name}   metric: {cfg['metric']}")
    print(f"Runtime: {result['seconds'] * 1000:.1f} ms")
    for rank, c in enumerate(result["candidates"], start=1):
        print(f"  {rank}. box={c['box']}  score={c['score']:.4f}")
    if not result["found"]:
        print("Verdict: Waldo NOT found (best score below threshold)")
    if truth is not None:
        rank = hit_rank(result["candidates"], truth, cfg.get("iou_threshold", 0.5))
        if rank is None:
            print(f"Truth {truth}: missed (not in top {cfg['top_k']})")
        else:
            print(f"Truth {truth}: found at rank {rank}")
    elif entry is not None:
        print("Truth: this crop has no Waldo")

    if not args.no_show:
        show("Where's Waldo (any key to close)", draw_result(image, result["candidates"], truth))


if __name__ == "__main__":
    main()
