# Where's Waldo Finder

A small, readable Waldo finder built on one idea: **template matching**.
Cut-outs of Waldo's head and hat are slid over the whole page at several
sizes, every position gets a similarity score, and the best-scoring spot is
the answer. No machine learning, one file to run, OpenCV and NumPy only.

![Scene 9 with the best guess boxed in red and a zoomed side panel](docs/example_9.jpg)

*Scene 9. Red = the finder's guess, green = the labeled answer, right panel =
the guess zoomed in with its score.*

**Contents:**
[Background](#background) ·
[Problem statement](#problem-statement) ·
[Approach](#approach) ·
[Results](#results) ·
[Quick start](#quick-start) ·
[Layout](#layout) ·
[Limits](#limits) ·
[Ethical implications](#ethical-implications) ·
[References](#references)

## Background

*Where's Waldo* is a visual search puzzle: a single small character hidden
in a page crammed with hundreds of look-alikes, stripes and red hats placed
there on purpose to fool you. That makes it a surprisingly good test case
for an image processing course project:

- **The target is well defined.** Waldo always has the same red-and-white
  striped shirt, round glasses and bobble hat, so a picture of him is a
  reasonable description of what to look for. No labeled dataset of
  thousands of examples is needed.
- **The scene is hard on purpose.** The pages are full of decoys at
  different sizes, partly hidden behind people and poles, so the naive
  version of every technique fails and each fix has to be reasoned about.
- **Every step can be looked at.** Unlike a trained neural network, each
  stage here (contrast, blur, mask, correlation, peaks) is a picture you can
  open and inspect, which is the point of learning image processing.

I chose this project because it lets a handful of classical tools
(histogram equalization, filtering, normalized cross-correlation, non-maximum
suppression) carry a whole, recognizable task from input image to answer,
and because it is easy to tell whether the answer is right.

## Problem statement

**Input:** one full *Where's Waldo* page as a color image, plus a few small
cut-outs of Waldo ("templates") taken from other pages.

**Output:** a single box on the input page marking where Waldo is.

**Correct** means the box overlaps the hand-labeled answer with an
intersection-over-union (IoU) of at least 0.5, the usual threshold in object
detection.

**Constraints I set myself:**

- Classical image processing only. No training, no neural networks.
- One file to run, understandable top to bottom in an afternoon.
- A full page should be searched in seconds on a laptop.
- It should work on pages the templates did not come from. A template cut
  from a page will always match that same page perfectly, so the honest
  test is to search a page using only templates from *other* pages.

## Approach

![Pipeline: template is masked, scene is preprocessed, both go through a loop over 17 sizes of masked ZNCC matching, peaks are merged, and the best guess is checked and shown](docs/pipeline.svg)

The finder has four parts. A longer, stage-by-stage version of the diagram
is in [docs/pipeline_detailed.svg](docs/pipeline_detailed.svg).

### 1. Template side: mask out what is not Waldo

The templates are cut straight out of a scene, so they contain things that
are not Waldo: background color in the corners and, in scene 1, a spear
across his face. `waldo/template_mask.py` builds a mask automatically:

- the dominant strong color along the template's border is treated as
  background and dropped;
- any tall, thin blob of strong color (a pole or spear) is dropped;
- everything else is kept and compared.

![The two templates, their masks, and what is actually compared](docs/templates.png)

*Left: template 1 with the yellow background and the spear masked out.
Right: template 9 has no background or pole, so the whole crop is kept.*

### 2. Scene side: even out the page

Color is kept, because the red and white stripes are the whole point.
Two small steps (`preprocess` in `waldo/matcher.py`) make pages drawn
with different inks comparable:

- **CLAHE** (contrast limited adaptive histogram equalization) stretches
  contrast separately in each small tile of the image, so a dim corner and
  a bright corner end up with similar contrast.
- A **5×5 Gaussian blur** softens the drawing so that tiny line differences
  between two artists' Waldos do not count against a match.

The template is given exactly the same treatment, so the two are compared
on equal terms.

### 3. The loop: slide the template at many sizes

Waldo is not drawn at the same size on every page, so the template is
resized from 0.6× to 1.4× in 17 steps. At each size:

- `cv2.matchTemplate` slides the masked template across the scene and
  scores every position with **ZNCC** (zero-mean normalized
  cross-correlation). ZNCC subtracts each window's mean and divides by its
  spread, so a match is about the *pattern* of stripes and glasses, not
  about brightness, and the mask keeps the dropped pixels out of the sum.
- The score map is **z-scored** (how many standard deviations each position
  sits above the average for that size) so that sizes can be compared
  fairly. A score of 4.5 means "4.5 standard deviations better than a
  typical spot".
- The **5 highest peaks** are kept, blanking a template-sized window around
  each one so they are not the same spot shifted by a pixel
  (non-maximum suppression).

### 4. Pick one

All candidates from all sizes are sorted by score and any box that overlaps
a better one (IoU above 0.3) is dropped. This runs once per template, and
the best box across all templates is the answer. `find_waldo.py` prints the
box and shows it in a window with a zoomed side panel.

### Why these choices

An earlier version of this repo (still on `main` at commit 3040bd5)
compared four correlation measures written from scratch (plain
cross-correlation, SSD, NCC, ZNCC), three color spaces, edge maps and a
red/white color filter, with a grid search over 240 combinations on small
crops. Findings that shaped the final design:

- Plain cross-correlation fails (68% top-5 accuracy) because any bright
  window scores high whether or not it looks like Waldo. Normalizing (NCC,
  ZNCC) fixes this.
- With a template from the *same* image, almost everything reaches 100%.
  With a template from the *other* image, the best setting was ZNCC on
  color with CLAHE and blur, at 70% top-5 on crops. Color plus smoothing is
  what helps when two Waldos are drawn slightly differently.
- On full pages, nothing worked until the template mask was added. The
  spear across Waldo's face in template 1 and the background in its corners
  were dragging every comparison down.

Everything that did not help was removed so the code stays small.

## Results

Two pages are labeled so far (1.jpg and 9.jpg), with one template cut from
each. Each page was searched two ways: with every template available, and
the honest "held-out" way, with only the template from the *other* page.

| Scene | Templates used | Found by | Score | Overlap with label | Result | Time |
|---|---|---|---|---|---|---|
| 1.jpg | all | 1.png (its own) | 8.35 | 1.00 | correct | 19.8 s |
| 1.jpg | held-out (9.png only) | 9.png | 5.74 | 0.76 | correct | 9.5 s |
| 9.jpg | all | 9.png (its own) | 12.58 | 1.00 | correct | 9.2 s |
| 9.jpg | held-out (1.png only) | 1.png | 4.53 | 0.76 | correct | 4.6 s |

Both pages are found as the #1 guess in both settings. The held-out rows
are the ones that matter: Waldo drawn by one page's template is found on a
page it has never seen, at a different size (0.80× and 1.35×) and with a
lower but still clearly-best score. Times are on a laptop CPU.

![Scene 1 with the best guess boxed in red and a zoomed side panel](docs/example_1.jpg)

*Scene 1, the harder page: Waldo is tiny, surrounded by hundreds of
red-and-blue soldiers, and has a spear in front of his face. The guess
(red) sits on top of the label (green).*

What the score means in practice: on scene 9 searched with the held-out
template, the winning spot scored 4.53 and the runner-up 4.15, so the margin
is real but not huge. A page with a better decoy could flip that order,
which is why the Limits section matters.

**Honest caveat.** Two labeled pages is a demonstration, not a benchmark.
The remaining 17 pages in `data/original` are unlabeled; labeling them (one
minute each with the labeler script) is the natural next step to get a real
accuracy number.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python find_waldo.py                       # searches the scene named at the top of the file
python find_waldo.py path/to/any_scene.jpg # or any image you like
```

A window opens with the result. Press any key to close it. Add `--no-show`
to print the box without a window.

`find_waldo.py` is the only file you run. It tries every template in
`data/templates` and keeps the single best-scoring box. When the scene is
one of the labeled pages it also prints `CORRECT` or `WRONG`.

## Layout

```
find_waldo.py             the one file to run
waldo/matcher.py          preprocessing, multi-scale ZNCC matching, box overlap
waldo/template_mask.py    mask out background and poles in the template
waldo/visualize.py        result window with the zoomed side panel
scripts/label_waldo.py    drag a box around Waldo in a new scene
scripts/make_templates.py cut templates out of the labeled scenes
data/original/            full pages (Hey-Waldo dataset, see data/original/SOURCE.md)
data/labels.json          Waldo's box in each labeled page
data/templates/           the cut-out templates, what the finder knows
docs/                     diagrams, the template strip and the example images
```

### Adding a page

```bash
python scripts/label_waldo.py 5.jpg      # drag a box around his head and hat
python scripts/make_templates.py         # writes data/templates/5.png
python find_waldo.py data/original/5.jpg # now searched with all templates
```

## Limits

Template matching finds things that look like the template. It struggles
when Waldo is mostly hidden, drawn from an unusual angle, or when another
striped figure is a closer match at some size. The search is also brute
force: 17 sizes × every pixel × every template, so time grows with page
size and template count. Labeling one more page and cutting its template
is the simplest fix if a new page fails: every template in the folder is
tried automatically.

## Ethical implications

This is a toy, but the technique is not. "Find one specific face or pattern
inside a crowded picture" is the same problem as locating a person in a
crowd photo or a surveillance frame. Template matching is far too weak to
do that reliably on real photographs, but the pipeline shape (describe the
target, scan the image at many sizes, rank the peaks) is exactly what
stronger detectors do, and the code here would not need to change much to
take a face crop instead of a Waldo crop. I think that is worth saying out
loud: the educational value of the project comes from the same idea that
underpins tools I would not want pointed at people without their consent.

Two smaller points:

- **Copyright.** The pages come from the Hey-Waldo dataset (ODbL licence),
  but the artwork itself belongs to its copyright holders. It is used here
  for study only and should not be redistributed as if it were free.
- **No personal data** is involved. The only images are published cartoon
  pages and the labels are boxes I drew.

## References

1. Hey-Waldo dataset, V. Constantinou:
   <https://github.com/vc1492a/Hey-Waldo> (folder `original-images`,
   ODbL-1.0). Source of all pages in `data/original`.
2. OpenCV template matching tutorial and `matchTemplate` reference:
   <https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html>,
   <https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html>
3. J. P. Lewis, "Fast Normalized Cross-Correlation", *Vision Interface*,
   1995. The standard reference for NCC/ZNCC template matching.
4. K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
   *Graphics Gems IV*, 1994, pp. 474–485. The CLAHE step.
5. R. Brunelli, *Template Matching Techniques in Computer Vision: Theory
   and Practice*, Wiley, 2009. Background on masked and multi-scale
   matching.
6. The earlier experiments this design was distilled from are in this
   repository's history on `main` at commit 3040bd5.
