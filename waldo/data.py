"""Paths and small helpers for reading the data files."""

import json
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"
TEMPLATE_DIR = DATA_DIR / "templates"
LABELS_PATH = DATA_DIR / "labels.json"


def read_image(path):
    """Read a color image (BGR, uint8) and fail loudly if it is missing."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def load_labels():
    """Return {image_name: [x, y, w, h]} for every labeled original image."""
    raw = load_json(LABELS_PATH)
    return {name: entry["waldo"] for name, entry in raw.items() if not name.startswith("_")}


def template_path(image_name):
    return TEMPLATE_DIR / f"{Path(image_name).stem}.png"
