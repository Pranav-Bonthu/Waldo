# Where's Waldo Finder

Finding Waldo with classical image processing: template matching with
correlation, plus color, filtering, histogram and edge techniques. Starting
small with zoomed-in crops so Waldo is easier to find, then scaling up.

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

```bash
# 1. (Optional) check or redraw Waldo's box in an original image
python scripts/label_waldo.py 1.jpg

# 2. Build templates + zoomed crops from data/labels.json
python scripts/make_dataset.py

# 3. Find Waldo in one crop (pop-up window: green = truth, red = guesses)
python scripts/find_waldo.py data/zoomed/1_pos0.png
python scripts/find_waldo.py data/zoomed/9_pos2.png --metric ncc --color-space bgr --blur

# 4. Try every combination and rank them
python scripts/grid_search.py
python scripts/grid_search.py --template-source other

# 5. Tests
python -m pytest
```

## Layout

```
config.json            default settings (each step can be toggled)
data/original/         full scenes (Hey-Waldo dataset, see SOURCE.md)
data/labels.json       Waldo's head + hat box in each labeled original
data/templates/        Waldo templates cut from the originals
data/zoomed/           zoomed crops (5x Waldo box) + manifest.json
waldo/preprocess.py    gray/BGR/HSV, global hist eq / CLAHE, Gaussian blur, Sobel / Canny
waldo/correlation.py   CCORR, SSD, NCC, ZNCC from scratch (loops + vectorized NumPy)
waldo/match.py         OpenCV or NumPy backend, higher score = better
waldo/peaks.py         top-k peaks with non-max suppression
waldo/color_mask.py    red/white filter with thresholds learned from the template
waldo/evaluate.py      IoU, rank of first correct guess, not-found threshold
scripts/               labeler, dataset builder, single run, grid search
results/               grid search tables (CSV)
```

## Current dataset

Images 1 and 9, five crops with Waldo and three without per image.
A guess is correct when its IoU with the true box is at least 0.5.

## First results

**Template from the same image** (240 combinations): 207 of 240 reach 100%
top-5 accuracy. SSD, NCC and ZNCC all hit 100%. Plain cross-correlation
averages 68% because bright windows score high whether or not they look like
Waldo. This case is easy since the template is an exact copy of the pixels.

**Template from the other image** (harder, more realistic): best is 70%
top-5 accuracy, from `ssd | bgr+blur` and `zncc | bgr+clahe+blur`. Color plus
smoothing helps when the two Waldos are drawn slightly differently.

## Next steps

- Multi-scale matching (image pyramid) so the template size doesn't have to match
- Mirror-flipped template
- FFT-based correlation
- Stripe-frequency detector
- More labeled images and bigger crops (less zoom)
