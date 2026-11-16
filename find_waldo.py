"""Find Waldo in a scene with template matching.

Usage:
    python find_waldo.py data/original/9.jpg --template data/templates/1.png
    python find_waldo.py data/original/1.jpg --template data/templates/9.png
    python find_waldo.py some_scene.jpg                  # uses data/templates/1.png

The window shows the scene with the #1 guess in a red box on the left
(green = true location, if the scene is labeled) and a zoomed crop of that
guess on the right. Press any key to close it.
"""

import argparse
from pathlib import Path

from waldo.data import ORIGINAL_DIR, TEMPLATE_DIR, load_labels, read_image
from waldo.evaluate import hit_rank
from waldo.matcher import SCALES, find_waldo_timed
from waldo.visualize import draw_side_panel, show


def truth_box(scene_path):
    """The labeled Waldo box if this scene is one of data/original, else None."""
    path = Path(scene_path).resolve()
    if path.parent != ORIGINAL_DIR.resolve():
        return None
    return load_labels().get(path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="image to search")
    parser.add_argument("--template", default=TEMPLATE_DIR / "1.png", help="Waldo template (head + hat)")
    parser.add_argument("--single-scale", action="store_true", help="match at one size only (faster, less robust)")
    parser.add_argument("--no-mask", action="store_true", help="use every template pixel, including background")
    parser.add_argument("--show-all", action="store_true", help="also print the other top-5 guesses")
    parser.add_argument("--no-show", action="store_true", help="print only, no window")
    args = parser.parse_args()

    scene = read_image(args.scene)
    template = read_image(args.template)
    candidates, seconds = find_waldo_timed(
        scene, template, scales=[1.0] if args.single_scale else SCALES, use_mask=not args.no_mask
    )

    print(f"Scene: {args.scene}   template: {Path(args.template).name}   time: {seconds:.1f} s")
    for rank, c in enumerate(candidates if args.show_all else candidates[:1], start=1):
        print(f"  {rank}. box={c['box']}  score={c['score']:.2f}  scale={c['scale']:.2f}")

    truth = truth_box(args.scene)
    if truth is not None:
        rank = hit_rank(candidates, truth)
        if rank == 1:
            print(f"Truth {truth}: #1 guess is CORRECT")
        elif rank is None:
            print(f"Truth {truth}: wrong, Waldo is not in the top {len(candidates)}")
        else:
            print(f"Truth {truth}: wrong, Waldo is guess #{rank}")

    if not args.no_show and candidates:
        show("Where's Waldo (any key to close)", draw_side_panel(scene, candidates[0], truth))


if __name__ == "__main__":
    main()
