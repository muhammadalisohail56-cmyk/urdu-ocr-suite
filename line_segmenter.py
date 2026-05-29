"""
line_segmenter.py
=================
A lightweight, dependency-free text-line detector for document pages.

UTRNet is a *line-level* recognizer (it was trained on single cropped lines,
height 32). Feeding it a whole page makes it read a heavily-downscaled image as
one line, which produces garbage. This module splits a page into individual line
crops first, so each crop is something UTRNet can actually read — turning the
local engine into a meaningful contributor rather than dead weight.

Approach: a classic **horizontal projection profile**.
    1. Binarize the page (Otsu) into an ink mask.
    2. Sum ink pixels per row -> a 1-D profile.
    3. Rows with enough ink are "text rows"; consecutive text rows form a band.
    4. Each band, padded slightly, is one line crop.

This is intentionally simple and fast. It works well on clean, well-spaced
printed documents (the UTRNet target domain). It is *not* a layout-analysis
engine: heavily skewed scans, multi-column layouts, or tightly-overlapping
Nastaliq ascenders/descenders can merge or split lines. For those, swap in an ML
line detector behind the same `segment_lines` interface.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

Box = Tuple[int, int, int, int]  # (left, top, right, bottom)


def _ink_mask(image: Image.Image, blur_radius: float, ink_delta: float) -> np.ndarray:
    """Local-background-subtraction binarization.

    Global thresholding (e.g. Otsu) fails on scanned/aged documents whose paper
    background is dark or unevenly lit — it ends up marking the paper itself as
    ink and saturating the profile. Instead we estimate the local paper
    background with a large Gaussian blur and mark a pixel as ink only when it is
    `ink_delta` gray levels *darker than its own local background*. This is
    robust to a gray or uneven background."""
    gray_img = image.convert("L")
    background = gray_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    gray = np.asarray(gray_img, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    return (bg - gray) > ink_delta


def _smooth(profile: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving-average smoothing to bridge tiny intra-line gaps."""
    if k <= 1:
        return profile
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(profile, kernel, mode="same")


def _bands_from_mask(text_rows: np.ndarray) -> List[Tuple[int, int]]:
    """Group consecutive True rows into (top, bottom) bands (bottom exclusive)."""
    bands: List[Tuple[int, int]] = []
    start = None
    for i, is_text in enumerate(text_rows):
        if is_text and start is None:
            start = i
        elif not is_text and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(text_rows)))
    return bands


def segment_lines(
    image: Image.Image,
    *,
    blur_radius: float = 30.0,
    ink_delta: float = 18.0,
    row_threshold_frac: float = 0.15,
    min_line_height: int = 8,
    pad: int = 4,
    smooth_k: int = 5,
) -> List[Tuple[Box, Image.Image]]:
    """Split a page image into top-to-bottom line crops.

    Parameters
    ----------
    blur_radius        : Gaussian radius for local-background estimation (px).
    ink_delta          : a pixel is ink if it is this many gray levels darker
                         than its local background.
    row_threshold_frac : a row counts as "text" if its (smoothed) ink count
                         exceeds this fraction of the profile's peak. Thresholding
                         *relative to the page's own peak* — rather than an
                         absolute count — is what separates lines on documents
                         where inter-line gaps still carry some ink (descenders,
                         noise, a page border).
    min_line_height    : bands shorter than this many pixels are discarded.
    pad                : vertical padding (px) added above/below each band.
    smooth_k           : moving-average window over the row profile.

    Returns a list of ``(box, crop)`` where ``box`` is ``(left, top, right,
    bottom)`` in the original image's coordinates. Returns an empty list if no
    lines are found (caller should fall back to the whole image).
    """
    ink = _ink_mask(image, blur_radius=blur_radius, ink_delta=ink_delta)
    h, w = ink.shape

    row_ink = _smooth(ink.sum(axis=1).astype(np.float32), k=smooth_k)
    # Peak via a high percentile so a single dark rule/header doesn't skew it.
    peak = float(np.percentile(row_ink, 99))
    if peak <= 0:
        return []
    threshold = row_threshold_frac * peak

    text_rows = row_ink > threshold
    lines: List[Tuple[Box, Image.Image]] = []
    for top, bottom in _bands_from_mask(text_rows):
        if (bottom - top) < min_line_height:
            continue
        t = max(0, top - pad)
        b = min(h, bottom + pad)
        box: Box = (0, t, w, b)
        lines.append((box, image.crop(box)))
    return lines
