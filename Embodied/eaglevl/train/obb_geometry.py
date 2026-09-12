from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Obb = Tuple[float, float, float, float, float]
QuantizedObb = Tuple[int, int, int, int, int]


def le90_canonicalize(w: float, h: float, theta_deg: float) -> tuple[float, float, float]:
    if h > w:
        w, h = h, w
        theta_deg = theta_deg + 90.0
    theta_deg = ((theta_deg + 90.0) % 180.0) - 90.0
    if theta_deg == 90.0:
        theta_deg = -90.0
    return float(w), float(h), float(theta_deg)


def theta_q_from_deg(theta_deg: float) -> int:
    q = int(round((theta_deg + 90.0) / 180.0 * 1000.0))
    return max(0, min(1000, q))


def theta_deg_from_q(theta_q: int) -> float:
    return 180.0 * float(theta_q) / 1000.0 - 90.0


def polygon_area(pts: np.ndarray) -> float:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _intersect_x(a: Sequence[float], b: Sequence[float], x: float) -> tuple[float, float]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    if dx == 0.0:
        return x, ay
    t = (x - ax) / dx
    return x, ay + t * (by - ay)


def _intersect_y(a: Sequence[float], b: Sequence[float], y: float) -> tuple[float, float]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dy = by - ay
    if dy == 0.0:
        return ax, y
    t = (y - ay) / dy
    return ax + t * (bx - ax), y


def _clip_poly(
    poly: list[tuple[float, float]],
    keep,
    intersect,
) -> list[tuple[float, float]]:
    if not poly:
        return poly
    out: list[tuple[float, float]] = []
    prev = poly[-1]
    prev_in = keep(prev)
    for cur in poly:
        cur_in = keep(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
        prev = cur
        prev_in = cur_in
    return out


def clip_polygon_to_rect(
    pts: np.ndarray,
    rect: tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    x1, y1, x2, y2 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    poly = [(float(p[0]), float(p[1])) for p in arr]
    poly = _clip_poly(poly, lambda p: p[0] >= x1, lambda a, b: _intersect_x(a, b, float(x1)))
    poly = _clip_poly(poly, lambda p: p[0] <= x2, lambda a, b: _intersect_x(a, b, float(x2)))
    poly = _clip_poly(poly, lambda p: p[1] >= y1, lambda a, b: _intersect_y(a, b, float(y1)))
    poly = _clip_poly(poly, lambda p: p[1] <= y2, lambda a, b: _intersect_y(a, b, float(y2)))
    if len(poly) < 3:
        return None
    clipped = np.asarray(poly, dtype=np.float64)
    if polygon_area(clipped) <= 0.0:
        return None
    return clipped


def quad_to_le90(pts: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(arr)
    w, h, theta = le90_canonicalize(float(w), float(h), float(angle))
    return float(cx), float(cy), w, h, theta


def quantize_obb(
    cx: float,
    cy: float,
    w: float,
    h: float,
    theta: float,
    tile_x: float,
    tile_y: float,
    tile_size: int,
) -> Optional[QuantizedObb]:
    if tile_size <= 0:
        return None
    w, h, theta = le90_canonicalize(w, h, theta)
    scale = 1000.0 / float(tile_size)
    cx_q = max(0, min(1000, int(round((cx - tile_x) * scale))))
    cy_q = max(0, min(1000, int(round((cy - tile_y) * scale))))
    w_q = max(0, min(1000, int(round(w * scale))))
    h_q = max(0, min(1000, int(round(h * scale))))
    if w_q == 0 or h_q == 0:
        return None
    return cx_q, cy_q, w_q, h_q, theta_q_from_deg(theta)


def dequantize_obb(
    cx_q: int,
    cy_q: int,
    w_q: int,
    h_q: int,
    th_q: int,
    tile_size: int,
    img_w: int,
    img_h: int,
) -> Obb:
    del img_w, img_h
    inv = float(tile_size) / 1000.0
    return (
        float(cx_q) * inv,
        float(cy_q) * inv,
        float(w_q) * inv,
        float(h_q) * inv,
        theta_deg_from_q(int(th_q)),
    )


def rotated_iou(obb_a: Obb, obb_b: Obb) -> float:
    wa, ha = float(obb_a[2]), float(obb_a[3])
    wb, hb = float(obb_b[2]), float(obb_b[3])
    if wa <= 0.0 or ha <= 0.0 or wb <= 0.0 or hb <= 0.0:
        return 0.0
    ra = ((float(obb_a[0]), float(obb_a[1])), (wa, ha), float(obb_a[4]))
    rb = ((float(obb_b[0]), float(obb_b[1])), (wb, hb), float(obb_b[4]))
    ret, region = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == 0 or region is None:
        return 0.0
    inter = float(cv2.contourArea(region))
    union = wa * ha + wb * hb - inter
    if union <= 0.0:
        return 0.0
    return inter / union
