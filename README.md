# Where's Waldo Finder

Finds Waldo in a full scene with template matching: a small picture of
Waldo's head and hat is slid over the whole page, and the spot that looks
most like it wins.

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python find_waldo.py data/original/9.jpg --template data/templates/1.png
python find_waldo.py data/original/1.jpg --template data/templates/9.png
```

A window opens with the scene on the left (red box = the finder's guess,
green box = the true spot when the scene is labeled) and a zoomed crop of
the guess on the right. Press any key to close it. A full scene takes about
10 to 20 seconds.

Flags: `--single-scale` (one template size, faster), `--no-mask` (use every
template pixel), `--show-all` (print the top 5 guesses), `--no-show`.

## How it works

![pipeline](docs/pipeline.svg)

A step-by-step version with every stage the images pass through is in
[docs/pipeline_detailed.svg](docs/pipeline_detailed.svg).

1. **Template mask** (`waldo/template_mask.py`). The template is cut straight
   out of a scene, so it contains things that are not Waldo: the background
   color at the corners and, in scene 1, a spear across his face. The mask
   marks those pixels so they do not count in the match.
2. **Preprocess** (`waldo/preprocess.py`). Scene and template get the same
   treatment: keep the color (the red and white stripes matter), CLAHE to
   even out contrast, Gaussian blur so two slightly different drawings of
   Waldo still line up.
3. **Match at many sizes** (`waldo/matcher.py`). Waldo is not drawn at the
   same size in every scene, so the template is resized from 0.6x to 1.4x in
   17 steps. At each size `cv2.matchTemplate` slides it over the scene and
   scores every position with zero-mean normalized cross-correlation (ZNCC).
4. **Pick the best peak**. Each size's score map is z-scored so sizes can be
   compared fairly, the top peaks are taken with non-max suppression, boxes
   that overlap a better one are dropped, and guess #1 is Waldo.

## Results

Template always from the other scene, so nothing is an exact pixel copy:

| Scene | Template | #1 guess correct |
|---|---|---|
| 1.jpg | 9.png | yes |
| 9.jpg | 1.png | yes |

Both scenes also pass with their own template. The tests
(`python -m pytest`) check the same thing on 10x crops so they run in seconds.

## Layout

```
find_waldo.py            run the finder
waldo/preprocess.py      CLAHE + blur
waldo/template_mask.py   mask out background and poles in the template
waldo/matcher.py         multi-scale ZNCC matching, peak picking
waldo/evaluate.py        IoU between boxes
waldo/visualize.py       result window with the zoomed side panel
waldo/data.py            paths, image reading, labels
scripts/label_waldo.py   drag a box around Waldo in a new scene
scripts/make_templates.py cut templates out of the labeled scenes
data/original/           full scenes (Hey-Waldo dataset, see SOURCE.md)
data/labels.json         Waldo's box in each labeled scene (1.jpg, 9.jpg)
data/templates/          the cut-out templates
tests/                   pytest
```

## Adding a scene

```bash
python scripts/label_waldo.py 5.jpg      # drag a box around his head and hat
python scripts/make_templates.py         # writes data/templates/5.png
```

## History

The earlier version of this repo compared four correlation metrics written
from scratch, several color spaces, edge maps and a red/white color filter
with a grid search. Only the combination above found Waldo in full scenes,
so the rest was removed to keep the code small. It is all still in the git
history on `main` before this branch (commit 3040bd5).
