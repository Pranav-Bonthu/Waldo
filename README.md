# Where's Waldo Finder

A small, readable Waldo finder built on one idea: **template matching**.
Cut-outs of Waldo's head and hat are slid over the whole page at several
sizes, every position gets a similarity score, and the best-scoring spot is
the answer. No machine learning, one file to run, OpenCV and NumPy only.

![Scene 9 with the best guess boxed in red and a zoomed side panel](docs/example_9.jpg)

*Scene 9 searched with the templates. Red = the finder's guess, green = the
labeled answer, right panel = the guess zoomed in.*

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python find_waldo.py                       # searches the scene named at the top of the file
python find_waldo.py path/to/any_scene.jpg # or any image you like
```

A window opens with the result. Press any key to close it. A full scene
takes 10 to 20 seconds. Add `--no-show` to print the box without a window.

`find_waldo.py` is the only file you run. It tries every template in
`data/templates` and keeps the single best-scoring box.

## How it works

![Pipeline: template is masked, scene is preprocessed, both go through a loop over 17 sizes of masked ZNCC matching, peaks are merged, and the best guess is checked and shown](docs/pipeline.svg)

**Template side** (`waldo/template_mask.py`). The template is cut straight
out of a scene, so it contains things that are not Waldo: background color
in the corners and, in scene 1, a spear across his face. The mask finds the
dominant strong color along the template border (background) and any tall,
thin blob of strong color (a pole), and tells the matcher to ignore those
pixels.

**Scene side** (`waldo/preprocess.py`). Color is kept, because the red and
white stripes are the whole point. CLAHE evens out contrast across the page
and a small Gaussian blur stops tiny drawing differences from mattering. The
template gets the exact same treatment so the two are comparable.

**The loop** (`waldo/matcher.py`). Waldo is not drawn at the same size in
every scene, so the template is resized from 0.6x to 1.4x in 17 steps. At
each size `cv2.matchTemplate` slides it across the scene and scores every
position with zero-mean normalized cross-correlation (ZNCC), which ignores
brightness differences and respects the mask. Each size's score map is
z-scored so sizes can be compared fairly, then the 5 highest peaks are
taken, blanking a template-sized window around each so they are not the
same spot shifted by a pixel.

**Picking one.** All candidates are sorted by score and any box that
overlaps a better one (IoU above 0.3) is dropped. This runs once per
template, and the best box across all templates is Waldo. When the scene is
labeled, the guess counts as correct if it overlaps the label with IoU of at
least 0.5.

A longer, stage-by-stage version of the diagram is in
[docs/pipeline_detailed.svg](docs/pipeline_detailed.svg).

## Results

Each labeled scene, searched with every template (so the other scene's
template is always in the mix):

| Scene | Guess | Time |
|---|---|---|
| 1.jpg | correct | ~20 s |
| 9.jpg | correct | ~10 s |

To check after a change, run the file on each labeled scene and read the
`CORRECT` line.

## Layout

```
find_waldo.py             the one file to run
waldo/matcher.py          preprocessing, multi-scale ZNCC matching, box overlap
waldo/template_mask.py    mask out background and poles in the template
waldo/visualize.py        result window with the zoomed side panel
scripts/label_waldo.py    drag a box around Waldo in a new scene
scripts/make_templates.py cut templates out of the labeled scenes
data/original/            full scenes (Hey-Waldo dataset, see SOURCE.md)
data/labels.json          Waldo's box in each labeled scene
data/templates/           the cut-out templates, what the finder knows
docs/                     diagrams and the example image
```

## Adding a scene

```bash
python scripts/label_waldo.py 5.jpg      # drag a box around his head and hat
python scripts/make_templates.py         # writes data/templates/5.png
python find_waldo.py data/original/5.jpg # now searched with all templates
```

## Limits

Template matching finds things that look like the template. It struggles
when Waldo is mostly hidden, drawn from an unusual angle, or when another
striped figure is a closer match at some size. Labeling one more scene and
cutting its template is the simplest fix if a new scene fails: every
template in the folder is tried automatically.

## History

The earlier version of this repo compared four correlation metrics written
from scratch, several color spaces, edge maps and a red/white color filter
with a grid search. Only the combination above found Waldo in full scenes,
so the rest was removed to keep the code small. It is all still in the git
history on `main` before this branch (commit 3040bd5).
