# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# -*- coding: utf-8 -*-
"""
Image augmentation module for training-time data augmentation.
"""

import random
from PIL import Image, ImageEnhance

from typing import Union, List, Tuple

# D4 extras (identity is the 0° JSONL). hflip_* = horizontal flip FIRST, then rotate.
GEOM_OPS: dict[str, tuple[int, bool, bool]] = {
    "rot90": (90, False, False),
    "rot180": (180, False, False),
    "rot270": (270, False, False),
    "hflip": (0, True, False),
    "hflip_rot90": (90, True, True),
    "hflip_rot180": (180, True, True),
    "hflip_rot270": (270, True, True),
}

_GEOM_ATOMS: dict[str, tuple[str, ...]] = {
    "rot90": ("rot90",),
    "rot180": ("rot180",),
    "rot270": ("rot270",),
    "hflip": ("hflip",),
    "hflip_rot90": ("hflip", "rot90"),
    "hflip_rot180": ("hflip", "rot180"),
    "hflip_rot270": ("hflip", "rot270"),
}


def resize_image_keep_aspect_ratio(
    image: Image.Image, 
    target_long_edge: int
) -> Image.Image:
    """
    Resize the image to the specified long edge length while preserving aspect ratio.
    
    Args:
        image: PIL Image
        target_long_edge: Target long edge length
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    long_edge = max(width, height)
    
    if long_edge == target_long_edge:
        return image
    
    scale = target_long_edge / long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def apply_resize_augmentation(
    image: Image.Image,
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> Image.Image:
    """
    Apply resize augmentation.
    
    Rules:
    - If data_augment=True, no processing is done and the original image is returned.
    - If data_augment=False, with augment_prob probability the original image is kept,
      and with (1-augment_prob) probability the long edge is resized to a random value
      in [min_long_edge, max_long_edge].
    
    Args:
        image: PIL Image
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image (default 50%)
    
    Returns:
        Processed PIL Image
    """
    if not data_augment:
        return image
    
    if random.random() < augment_prob:
        return image
    
    width, height = image.size
    current_long_edge = max(width, height)
    
    target_long_edge = random.randint(min_long_edge, max_long_edge)
    
    if target_long_edge == current_long_edge:
        return image
    
    return resize_image_keep_aspect_ratio(image, target_long_edge)


def apply_resize_augmentation_to_list(
    images: List[Image.Image],
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> List[Image.Image]:
    """
    Apply resize augmentation to a list of images.
    
    Note: Each image independently decides whether to augment and the target size.
    
    Args:
        images: List of PIL Images
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image
    
    Returns:
        List of processed PIL Images
    """
    return [
        apply_resize_augmentation(
            img, 
            data_augment=data_augment,
            min_long_edge=min_long_edge,
            max_long_edge=max_long_edge,
            augment_prob=augment_prob
        )
        for img in images
    ]


def transform_xy(x: int, y: int, op: str) -> tuple:
    """Map one normalized [0, 1000] coordinate through a geometry op.

    Ops match the PIL transposes applied by `apply_image_geometry`:
      rot90  = Transpose.ROTATE_90  (90° CCW): (x, y) -> (y, 1000 - x)
      rot180 = Transpose.ROTATE_180:           (x, y) -> (1000 - x, 1000 - y)
      rot270 = Transpose.ROTATE_270 (90° CW):  (x, y) -> (1000 - y, x)
      hflip  = Transpose.FLIP_LEFT_RIGHT:      (x, y) -> (1000 - x, y)
    """
    if op == "rot90":
        return (y, 1000 - x)
    if op == "rot180":
        return (1000 - x, 1000 - y)
    if op == "rot270":
        return (1000 - y, x)
    if op == "hflip":
        return (1000 - x, y)
    raise ValueError(f"Unknown geometry op {op!r}; expected rot90/rot180/rot270/hflip")


def transform_box(x1: int, y1: int, x2: int, y2: int, op: str) -> tuple:
    """Transform an axis-aligned box in [0, 1000] token space.

    Maps all four corners through `transform_xy`, re-normalizes to
    (min, max) per axis, clamps to [0, 1000], and nudges degenerate
    edges by +1 so the box keeps nonzero extent.
    """
    corners = [
        transform_xy(cx, cy, op)
        for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    nx1, nx2 = max(0, min(1000, min(xs))), max(0, min(1000, max(xs)))
    ny1, ny2 = max(0, min(1000, min(ys))), max(0, min(1000, max(ys)))
    if nx2 == nx1:
        nx2 = min(1000, nx1 + 1)
    if ny2 == ny1:
        ny2 = min(1000, ny1 + 1)
    return (nx1, ny1, nx2, ny2)

def resolve_geom_op(op: str) -> tuple[int, bool, bool]:
    if op not in GEOM_OPS:
        raise ValueError(f"Unknown geom_op {op!r}; expected one of {sorted(GEOM_OPS)}")
    return GEOM_OPS[op]


def geom_op_atoms(op: str) -> tuple[str, ...]:
    if op not in _GEOM_ATOMS:
        raise ValueError(f"Unknown geom_op {op!r}; expected one of {sorted(_GEOM_ATOMS)}")
    return _GEOM_ATOMS[op]


def transform_obb(
    cx: int, cy: int, w: int, h: int, th_q: int, op: str
) -> tuple[int, int, int, int, int]:
    """Transform a LE90 OBB in [0, 1000] token space through a D4 geom_op."""
    from eaglevl.train.obb_geometry import le90_canonicalize, theta_deg_from_q, theta_q_from_deg

    ncx, ncy = int(cx), int(cy)
    wf, hf = float(w), float(h)
    theta = theta_deg_from_q(int(th_q))
    for atom in geom_op_atoms(op):
        ncx, ncy = transform_xy(ncx, ncy, atom)
        if atom == "hflip":
            wf, hf, theta = le90_canonicalize(wf, hf, -theta)
        elif atom == "rot90":
            wf, hf, theta = le90_canonicalize(wf, hf, theta + 90.0)
        elif atom == "rot180":
            wf, hf, theta = le90_canonicalize(wf, hf, theta + 180.0)
        elif atom == "rot270":
            wf, hf, theta = le90_canonicalize(wf, hf, theta + 270.0)
        else:
            raise ValueError(f"Unknown geometry atom {atom!r}")
    ncx = max(0, min(1000, int(ncx)))
    ncy = max(0, min(1000, int(ncy)))
    wf, hf, theta = le90_canonicalize(wf, hf, theta)
    w_q = max(1, min(1000, int(round(wf))))
    h_q = max(1, min(1000, int(round(hf))))
    if h_q > w_q:
        w_q, h_q, theta = le90_canonicalize(float(w_q), float(h_q), theta)
        w_q = max(1, min(1000, int(round(w_q))))
        h_q = max(1, min(1000, int(round(h_q))))
    return ncx, ncy, int(w_q), int(h_q), theta_q_from_deg(theta)

def apply_image_geometry(
    image: Image.Image,
    rotate: int = 0,
    hflip: bool = False,
    flip_first: bool = False,
) -> Image.Image:
    """Rotate/flip a PIL image to match a token-space geometry op.

    `rotate` must be 0, 90, 180, or 270 (degrees CCW, matching
    Image.Transpose.ROTATE_*). Default order is rotate then hflip (HBB keys).
    `flip_first=True` applies hflip before rotate (OBB D4 `hflip_rot*` keys).
    With rotate=0 and hflip=False the same image object is returned.
    """
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be one of 0|90|180|270, got {rotate!r}")

    def _rotate(img: Image.Image) -> Image.Image:
        if rotate == 0:
            return img
        return img.transpose(
            {
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270,
            }[rotate]
        )

    def _hflip(img: Image.Image) -> Image.Image:
        if not hflip:
            return img
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if flip_first:
        return _rotate(_hflip(image))
    return _hflip(_rotate(image))


def apply_color_jitter(
    image: Image.Image,
    enabled: bool = False,
    strength: float = 0.2,
) -> Image.Image:
    """Photometric jitter: brightness, contrast, saturation (no hue).

    Each factor is drawn from uniform(1 - strength, 1 + strength).
    Returns the input object unchanged when disabled. Uses the
    module-level `random`, which the training dataset seeds per index.
    """
    if not enabled:
        return image
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = random.uniform(1 - strength, 1 + strength)
        image = enhancer(image).enhance(factor)
    return image
