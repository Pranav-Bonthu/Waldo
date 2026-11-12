"""Run every pre-processing + metric combination on the zoomed crops.

Usage:
    python scripts/grid_search.py
    python scripts/grid_search.py --template-source other   # harder: template from a different image
    python scripts/grid_search.py --no-show

For each combination it reports:
    top5      fraction of Waldo crops where a correct box (IoU >= 0.5) is in the top 5
    neg_rej   fraction of no-Waldo crops correctly rejected, using the best
              'not found' score threshold for that combination
The full table is saved to results/grid_results.csv and the top rows are
printed. At the end, the best combination is shown on one crop in a pop-up.
"""

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waldo.data import MANIFEST_PATH, REPO_ROOT, TEMPLATE_DIR, ZOOMED_DIR, load_config, load_json, read_image  # noqa: E402
from waldo.detector import detect  # noqa: E402
from waldo.evaluate import best_threshold, hit_rank  # noqa: E402
from waldo.visualize import draw_result, show  # noqa: E402

COLOR_SPACES = ["gray", "bgr", "hsv"]
HIST_EQ = ["none", "global", "clahe"]
SMOOTHING = [False, True]
EDGES = ["none", "sobel", "canny"]
METRICS = ["ccorr", "ssd", "ncc", "zncc"]
COLOR_FILTER = [False, True]


def combinations():
    """All valid combinations. Edge maps are gray only, so skip color + edges."""
    for cs, he, sm, ed, me, cf in itertools.product(COLOR_SPACES, HIST_EQ, SMOOTHING, EDGES, METRICS, COLOR_FILTER):
        if ed != "none" and cs != "gray":
            continue
        yield {"color_space": cs, "hist_eq": he, "smoothing": sm, "edges": ed, "metric": me, "color_filter": cf}


def make_cfg(base, combo):
    cfg = load_config()
    cfg.update(base)
    cfg["color_space"] = combo["color_space"]
    cfg["hist_eq"] = combo["hist_eq"]
    cfg["smoothing"] = dict(cfg["smoothing"], enabled=combo["smoothing"])
    cfg["edges"] = combo["edges"]
    cfg["metric"] = combo["metric"]
    cfg["color_filter"] = dict(cfg["color_filter"], enabled=combo["color_filter"])
    cfg["not_found_threshold"] = None
    return cfg


def combo_name(combo):
    parts = [combo["color_space"]]
    if combo["hist_eq"] != "none":
        parts.append(combo["hist_eq"])
    if combo["smoothing"]:
        parts.append("blur")
    if combo["edges"] != "none":
        parts.append(combo["edges"])
    if combo["color_filter"]:
        parts.append("colorfilter")
    return f"{combo['metric']:<5} | " + "+".join(parts)


def pick_template(entry, template_source, all_templates):
    if template_source == "same":
        return entry["template"]
    others = [t for t in all_templates if t != entry["template"]]
    return others[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template-source", choices=["same", "other"], default="same")
    parser.add_argument("--rows", type=int, default=20, help="how many top rows to print")
    parser.add_argument("--no-show", action="store_true", help="skip the pop-up of the best result")
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    crops = manifest["crops"]
    all_templates = sorted({c["template"] for c in crops})
    images = {c["file"]: read_image(ZOOMED_DIR / c["file"]) for c in crops}
    templates = {t: read_image(TEMPLATE_DIR / t) for t in all_templates}

    base = load_config()
    combos = list(combinations())
    print(f"{len(combos)} combinations x {len(crops)} crops (template source: {args.template_source})")

    rows = []
    start = time.perf_counter()
    for combo in combos:
        cfg = make_cfg(base, combo)
        hits = 0
        n_pos = 0
        pos_scores = []
        neg_scores = []
        for entry in crops:
            tpl = templates[pick_template(entry, args.template_source, all_templates)]
            result = detect(images[entry["file"]], tpl, cfg)
            if entry["box"] is None:
                neg_scores.append(result["best_score"])
                continue
            n_pos += 1
            pos_scores.append(result["best_score"])
            if hit_rank(result["candidates"], entry["box"], cfg["iou_threshold"]) is not None:
                hits += 1
        threshold, neg_rej = best_threshold(pos_scores, neg_scores)
        rows.append({
            "combo": combo_name(combo),
            "top5": hits / n_pos,
            "neg_rej": neg_rej,
            "threshold": threshold,
            **{k: v for k, v in combo.items()},
        })

    rows.sort(key=lambda r: (-r["top5"], -(r["neg_rej"] or 0.0)))

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"grid_results_{args.template_source}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done in {time.perf_counter() - start:.1f} s. Full table: {csv_path.relative_to(REPO_ROOT)}\n")
    print(f"{'rank':>4}  {'metric | pre-processing':<40} {'top5':>6} {'neg_rej':>8}")
    for i, r in enumerate(rows[:args.rows], start=1):
        neg = "n/a" if r["neg_rej"] is None else f"{r['neg_rej']:.2f}"
        print(f"{i:>4}  {r['combo']:<40} {r['top5']:>6.2f} {neg:>8}")

    print("\nTop-5 accuracy by metric (best combination for each):")
    for metric in METRICS:
        best = max((r for r in rows if r["metric"] == metric), key=lambda r: r["top5"])
        print(f"  {metric:<5} {best['top5']:.2f}  ({best['combo'].split('| ')[1]})")

    if not args.no_show:
        best = rows[0]
        cfg = make_cfg(base, {k: best[k] for k in ["color_space", "hist_eq", "smoothing", "edges", "metric", "color_filter"]})
        entry = next(c for c in crops if c["box"] is not None)
        tpl = templates[pick_template(entry, args.template_source, all_templates)]
        result = detect(images[entry["file"]], tpl, cfg)
        show(f"Best: {best['combo']} on {entry['file']}", draw_result(images[entry["file"]], result["candidates"], entry["box"]))


if __name__ == "__main__":
    main()
