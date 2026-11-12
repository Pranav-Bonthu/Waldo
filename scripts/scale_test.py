"""Test how each method holds up as the search area gets bigger.

Usage:
    python scripts/scale_test.py
    python scripts/scale_test.py --template-source other
    python scripts/scale_test.py --levels 5 10 20 full --positives 5 --negatives 3
    python scripts/scale_test.py --no-show

Zoom levels:
    5, 10, 20   square crops whose side is that many times Waldo's box size,
                with Waldo at a random spot (plus crops without Waldo)
    full        the whole original scene

Crops are made in memory with a fixed seed, so runs are repeatable and
nothing is written to data/. For each method and level it prints top-5
accuracy: the fraction of Waldo images where a correct box (IoU >= 0.5)
is among the 5 guesses. It also prints how often the no-Waldo crops are
rejected, using the best 'not found' threshold for that method and level.
At the end it shows the best method on a full scene in a pop-up.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waldo.crops import negative_crop, positive_crop  # noqa: E402
from waldo.data import ORIGINAL_DIR, REPO_ROOT, load_config, load_labels, read_image  # noqa: E402
from waldo.detector import detect  # noqa: E402
from waldo.evaluate import best_threshold, hit_rank  # noqa: E402
from waldo.visualize import draw_result, show  # noqa: E402

# A short list of methods chosen from the zoomed-in grid search: the plain
# baseline for each metric, plus the combinations that did best when the
# template came from a different image.
METHODS = [
    {"metric": "ccorr", "color_space": "gray"},
    {"metric": "ssd", "color_space": "gray"},
    {"metric": "ncc", "color_space": "gray"},
    {"metric": "zncc", "color_space": "gray"},
    {"metric": "zncc", "color_space": "bgr"},
    {"metric": "ssd", "color_space": "bgr", "blur": True},
    {"metric": "zncc", "color_space": "bgr", "hist_eq": "clahe", "blur": True},
    {"metric": "zncc", "color_space": "bgr", "hist_eq": "clahe", "blur": True, "color_filter": True},
    {"metric": "zncc", "color_space": "gray", "color_filter": True},
    {"metric": "zncc", "color_space": "gray", "edges": "sobel"},
    {"metric": "ncc", "color_space": "gray", "hist_eq": "global", "edges": "canny"},
]


def method_name(m):
    parts = [m["color_space"]]
    if m.get("hist_eq", "none") != "none":
        parts.append(m["hist_eq"])
    if m.get("blur"):
        parts.append("blur")
    if m.get("edges", "none") != "none":
        parts.append(m["edges"])
    if m.get("color_filter"):
        parts.append("colorfilter")
    return f"{m['metric']:<5} | " + "+".join(parts)


def make_cfg(m):
    cfg = load_config()
    cfg["metric"] = m["metric"]
    cfg["color_space"] = m["color_space"]
    cfg["hist_eq"] = m.get("hist_eq", "none")
    cfg["edges"] = m.get("edges", "none")
    cfg["smoothing"] = dict(cfg["smoothing"], enabled=m.get("blur", False))
    cfg["color_filter"] = dict(cfg["color_filter"], enabled=m.get("color_filter", False))
    cfg["multiscale"] = dict(cfg["multiscale"], enabled=m.get("multiscale", False))
    cfg["template_mask"] = dict(cfg["template_mask"], enabled=m.get("mask", False))
    cfg["not_found_threshold"] = None
    return cfg


def build_tests(level, originals, labels, n_pos, n_neg, seed):
    """List of (name, source_image_name, image, truth_box_or_None)."""
    tests = []
    for name in sorted(labels):
        image = originals[name]
        box = labels[name]
        stem = Path(name).stem
        if level == "full":
            tests.append((f"{stem}_full", name, image, box))
            continue
        rng = random.Random(f"{seed}-{level}-{name}")
        size = int(round(float(level) * max(box[2], box[3])))
        for i in range(n_pos):
            cx, cy, sw, sh = positive_crop(rng, image.shape, box, size)
            crop = image[cy:cy + sh, cx:cx + sw]
            truth = [box[0] - cx, box[1] - cy, box[2], box[3]]
            tests.append((f"{stem}_z{level}_pos{i}", name, crop, truth))
        for i in range(n_neg):
            corner = negative_crop(rng, image.shape, box, size)
            if corner is None:
                break
            cx, cy = corner
            crop = image[cy:cy + size, cx:cx + size]
            tests.append((f"{stem}_z{level}_neg{i}", name, crop, None))
    return tests


def template_for(source_name, template_source, templates):
    if template_source == "same":
        return templates[source_name]
    others = [n for n in sorted(templates) if n != source_name]
    return templates[others[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", nargs="+", default=["5", "10", "20", "full"])
    parser.add_argument("--template-source", choices=["same", "other"], default="same")
    parser.add_argument("--positives", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--multiscale", action="store_true", help="try the template at many sizes")
    parser.add_argument("--mask", action="store_true", help="ignore template pixels that are not Waldo")
    parser.add_argument("--methods", type=int, nargs="+", help="only run these method numbers (1-based)")
    args = parser.parse_args()

    labels = load_labels()
    originals = {name: read_image(ORIGINAL_DIR / name) for name in labels}
    templates = {}
    for name, (x, y, w, h) in labels.items():
        templates[name] = originals[name][y:y + h, x:x + w].copy()

    global METHODS
    if args.methods:
        METHODS = [METHODS[i - 1] for i in args.methods]
    for m in METHODS:
        m["multiscale"] = args.multiscale
        m["mask"] = args.mask

    rows = []
    timing = {}
    for level in args.levels:
        tests = build_tests(level, originals, labels, args.positives, args.negatives, args.seed)
        n_pos = sum(1 for t in tests if t[3] is not None)
        n_neg = len(tests) - n_pos
        example = tests[0][2].shape
        print(f"Level {level:>4}: {n_pos} with Waldo, {n_neg} without, e.g. {example[1]}x{example[0]} px")
        start = time.perf_counter()
        for m in METHODS:
            cfg = make_cfg(m)
            hits = 0
            pos_scores = []
            neg_scores = []
            for _, source, image, truth in tests:
                result = detect(image, template_for(source, args.template_source, templates), cfg)
                if truth is None:
                    neg_scores.append(result["best_score"])
                    continue
                pos_scores.append(result["best_score"])
                if hit_rank(result["candidates"], truth, cfg["iou_threshold"]) is not None:
                    hits += 1
            _, neg_rej = best_threshold(pos_scores, neg_scores)
            rows.append({"method": method_name(m), "level": level, "top5": hits / n_pos, "neg_rej": neg_rej})
        timing[level] = (time.perf_counter() - start) / (len(METHODS) * len(tests))

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = ""
    if args.multiscale:
        suffix += "_multiscale"
    if args.mask:
        suffix += "_mask"
    if args.methods or args.levels != ["5", "10", "20", "full"]:
        suffix += "_partial"
    csv_path = results_dir / f"scale_results_{args.template_source}{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "level", "top5", "neg_rej"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTop-5 accuracy (template source: {args.template_source}, multiscale: {args.multiscale}, mask: {args.mask})")
    header = f"{'metric | pre-processing':<40}" + "".join(f"{lv:>7}" for lv in args.levels)
    print(header)
    for m in METHODS:
        name = method_name(m)
        cells = []
        for level in args.levels:
            row = next(r for r in rows if r["method"] == name and r["level"] == level)
            cells.append(f"{row['top5']:>7.2f}")
        print(f"{name:<40}" + "".join(cells))
    print("avg ms per image" + " " * 24 + "".join(f"{timing[lv] * 1000:>7.0f}" for lv in args.levels))
    print(f"\nFull table incl. no-Waldo rejection: {csv_path.relative_to(REPO_ROOT)}")

    if not args.no_show:
        last = args.levels[-1]
        best = max((r for r in rows if r["level"] == last), key=lambda r: r["top5"])
        m = next(m for m in METHODS if method_name(m) == best["method"])
        name = sorted(labels)[0]
        image = originals[name]
        result = detect(image, template_for(name, args.template_source, templates), make_cfg(m))
        show(f"Best at {last}: {best['method']} on {name}", draw_result(image, result["candidates"], labels[name]))


if __name__ == "__main__":
    main()
