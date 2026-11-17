# Where's Waldo Finder

A small Waldo finder built on one idea: slide a picture of Waldo over the
whole page, score how well it fits at every spot, and box the best one.
No machine learning. One file to run. OpenCV and NumPy only.

![Page 9 with the best guess boxed in red and a zoomed side panel](docs/example_9.jpg)

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

*Where's Wally?* is a series of picture books by the British illustrator
Martin Handford. Every page is a crowded drawing of hundreds of
tiny people at a beach, a train station, a battlefield, and somewhere in
the crowd is Waldo: a thin man in a red and white striped shirt, a red and
white bobble hat and round glasses. The game is to find him.

The books are designed to make that hard. Waldo is drawn very small, often
under half an inch tall. The crowd is full of people in red and white
clothes, striped shirts and red hats. He is usually half hidden behind
someone, a pole or the edge of a tent. People can spend minutes on one
page.

## Problem statement

Can a program find Waldo on a page it has never seen?

Given a full page and a small cut-out of Waldo taken from a different
page, draw one box around Waldo. The box is right if it lands on him,
which we define as overlapping the hand-drawn answer box by at least half
(the usual rule in object detection, called intersection over union or
IoU).

## Approach

![Eight steps from a Waldo cut-out and a page to a red box, with a bubble explaining each step](docs/pipeline.svg)

The code for steps 2 to 7 is in `waldo/`, and
`find_waldo.py` runs them in order. A more technical version of the
diagram, with the OpenCV names, is in
[docs/pipeline_detailed.svg](docs/pipeline_detailed.svg).

**1. What we start with.** A small cut-out of Waldo's head and hat
(the "template") and the page to search. Both stay in color because the
red and white stripes are the main clue.

**2. Clean up the cut-out.** The cut-out was snipped straight out of a
page, so it has things in it that are not Waldo: yellow background in the
corners and, on page 1, a spear across his face. The program finds the
main color along the cut-out's edge and treats it as background, and finds
any tall thin streak of strong color and treats it as a pole. Those pixels
get a "skip me" mark called a mask. The picture below shows the two
cut-outs, their masks and what is left to compare.

![The two cut-outs, their masks, and what is actually compared](docs/templates.png)

*Left: cut-out 1 with the yellow background and the spear masked out.
Right: cut-out 9 has neither, so all of it is kept.*

**3. Even out the page.** Two pages drawn with different inks or scanned
differently should still be comparable, so the page gets two small fixes.
First the contrast is evened out separately in each small tile of the
image, so a dim corner and a bright corner end up looking alike. The
technique is called CLAHE, short for contrast limited adaptive histogram
equalization; "adaptive" means per tile, "limited" means it will not
over-boost flat areas. Then a light blur (5 by 5 pixels) softens the
drawing so tiny pen strokes do not count. The cut-out gets the same
two fixes so the two images are on equal terms.

**4. Try many sizes.** Waldo is bigger on some pages than others. The
cut-out is resized from 0.6 times to 1.4 times its original size in 17
steps, and each size is tried in turn.

**5. Slide and score.** At each size the cut-out is laid over every
position on the page and compared with the patch underneath it. The
comparison first subtracts the average brightness of each patch and
divides by how spread out its values are, so a match is about the pattern
of stripes, glasses and hat and not about how light or dark the area is.
Masked pixels are left out of the sum. This measure is called zero-mean
normalized cross-correlation (ZNCC). OpenCV computes it for every position
at once, so the result is a score map the size of the page.

**6. Keep the best spots.** Score maps from different sizes are not on the
same scale, so each map is converted to "how many standard deviations
above this map's average". A score of 4.5 means the spot is 4.5 standard
deviations better than a typical spot on that page. The 5 highest spots
per size are kept. After each one is taken, a cut-out-sized window around
it is blanked out so the next pick is a different place and not the same
spot shifted by a pixel.

**7. Pick one.** All spots from all sizes and all cut-outs are sorted by
score. Any spot that overlaps a better one is dropped. The top spot is
the answer: a box, a score and the size it was found at.

**8. Show it.** The page opens in a window with the guess in red and, if
the page has a hand-drawn answer, that box in green. The right side of the
window is the guess zoomed in. The program also prints the box, the score
and whether the guess was right.

## Layout

```
find_waldo.py             the one file to run
waldo/matcher.py          evening out, sliding at many sizes, scoring, box overlap
waldo/template_mask.py    mask out background and poles in the cut-out
waldo/visualize.py        result window with the zoomed side panel
scripts/label_waldo.py    drag a box around Waldo on a new page
scripts/make_templates.py cut templates out of the labeled pages
data/original/            full pages (Hey-Waldo dataset, see data/original/SOURCE.md)
data/labels.json          Waldo's box on each labeled page
data/templates/           the cut-outs, what the finder knows
docs/                     diagrams, the cut-out strip and the example images
```

### Adding a page

```bash
python scripts/label_waldo.py 5.jpg      # drag a box around his head and hat
python scripts/make_templates.py         # writes data/templates/5.png
python find_waldo.py data/original/5.jpg # now searched with all cut-outs
```

## Limits

The finder looks for things that look like the cut-out. It struggles when
Waldo is mostly hidden, drawn from an unusual angle, or when another
striped figure is a closer match at some size. The search is also brute
force, 17 sizes times every pixel times every cut-out, so time grows with
page size and the number of cut-outs. 

## Ethical implications

Finding one small figure in a crowded picture is the same problem as
finding one person in a crowd photo or a security camera frame. Template
matching is far too weak to do that on real photographs, but the shape of
the pipeline (describe the target, scan the image at many sizes, rank the
spots) is what real detectors do, and swapping the Waldo cut-out for a
face crop would take very little code. That is the use I would not want
this to feed: tracking people who have not agreed to be tracked. A course
project will not change that either way, but it is where this idea leads.

Copyright is the other point. The pages come from the Hey-Waldo dataset,
which is under the Open Database License, but the drawings themselves
belong to their copyright holders. They are used here for study and
should not be passed on as if they were free.
