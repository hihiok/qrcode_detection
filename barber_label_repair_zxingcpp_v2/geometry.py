from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


FIELD_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


def quad_geometry_issue(points: np.ndarray, min_span: float = 4.0) -> str | None:
    """Return a stable reason when a manual quad cannot define a usable 2-D ROI."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2):
        return "shape_not_4x2"
    if not np.isfinite(p).all():
        return "non_finite_coordinate"
    if len(np.unique(np.round(p, 6), axis=0)) != 4:
        return "duplicate_vertices"
    span = p.max(axis=0) - p.min(axis=0)
    if span[0] < min_span:
        return "bbox_width_lt_4px"
    if span[1] < min_span:
        return "bbox_height_lt_4px"
    idx = geometric_order(p)
    q = p[idx]
    area2 = abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) -
                             np.roll(q[:, 0], -1) * q[:, 1])))
    if area2 < 2.0:
        return "polygon_area_lt_1px2"
    return None


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    hp = np.c_[points, np.ones(len(points))] @ np.asarray(h, dtype=np.float64).T
    if np.any(np.abs(hp[:, 2]) < 1e-12):
        raise ValueError("homogeneous point at infinity")
    return hp[:, :2] / hp[:, 2:3]


def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("homography requires two (4,2) arrays")
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y],
              [0, 0, 0, x, y, 1, -v * x, -v * y]]
    b = dst.reshape(-1)
    h = np.linalg.solve(np.asarray(a), b)
    return np.r_[h, 1.0].reshape(3, 3)


def geometric_order(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2):
        raise ValueError("quad must be (4,2)")
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    cyc = list(map(int, order))
    start = min(range(4), key=lambda i: (p[cyc[i], 0] + p[cyc[i], 1],
                                         p[cyc[i], 1], p[cyc[i], 0]))
    cyc = cyc[start:] + cyc[:start]
    q = p[cyc]
    area2 = float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) -
                         np.roll(q[:, 0], -1) * q[:, 1]))
    if area2 < 0:
        cyc = [cyc[0], cyc[3], cyc[2], cyc[1]]
    return np.asarray(cyc, dtype=np.int64)


def rotation_matrix_ccw(k: int, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    k %= 4
    if k == 0:
        return np.eye(3), (width, height)
    if k == 1:
        return np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1.0]]), (height, width)
    if k == 2:
        return np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1.0]]), (width, height)
    return np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1.0]]), (height, width)


def rotate_image_and_h(image: Image.Image, h: np.ndarray, k: int) -> tuple[Image.Image, np.ndarray]:
    k %= 4
    if k == 0:
        return image, h
    method = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
              3: Image.Transpose.ROTATE_270}[k]
    r, _ = rotation_matrix_ccw(k, *image.size)
    return image.transpose(method), r @ h


@dataclass(frozen=True)
class View:
    image: Image.Image
    original_to_view: np.ndarray
    geometry_family: str
    geometry_id: str


def crop_view(image: Image.Image, quad: np.ndarray, margin: float, target: int = 512) -> View:
    p = np.asarray(quad, dtype=np.float64)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    left = max(0, int(np.floor(x0 - dx)))
    top = max(0, int(np.floor(y0 - dy)))
    right = min(image.width, int(np.ceil(x1 + dx)) + 1)
    bottom = min(image.height, int(np.ceil(y1 + dy)) + 1)
    if right - left < 4 or bottom - top < 4:
        raise ValueError("degenerate crop")
    scale = target / max(right - left, bottom - top)
    out_size = (max(4, int(round((right - left) * scale))),
                max(4, int(round((bottom - top) * scale))))
    out = image.crop((left, top, right, bottom)).resize(out_size, Image.Resampling.LANCZOS)
    h = np.array([[scale, 0, -left * scale], [0, scale, -top * scale], [0, 0, 1.0]])
    return View(out, h, "crop", f"crop_m{margin:.2f}_t{target}")


def warp_view(image: Image.Image, quad: np.ndarray, size: int, quiet: float) -> View:
    idx = geometric_order(quad)
    src = np.asarray(quad, dtype=np.float64)[idx]
    q = float(size) * quiet
    dst = np.array([[q, q], [size - 1 - q, q], [size - 1 - q, size - 1 - q],
                    [q, size - 1 - q]], dtype=np.float64)
    h = solve_homography(src, dst)
    inv = np.linalg.inv(h)
    inv /= inv[2, 2]
    coeffs = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1],
              inv[1, 2], inv[2, 0], inv[2, 1])
    out = image.transform((size, size), Image.Transform.PERSPECTIVE, coeffs,
                          Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    return View(out, h, "warp", f"warp_s{size}_q{quiet:.2f}")


def best_vertex_assignment(detected: np.ndarray, manual: np.ndarray) -> tuple[tuple[int, ...], float, float]:
    d = np.asarray(detected, dtype=np.float64)
    m = np.asarray(manual, dtype=np.float64)
    diag = max(1.0, float(np.linalg.norm(m.max(axis=0) - m.min(axis=0))))
    best = None
    for perm in itertools.permutations(range(4)):
        errs = np.linalg.norm(d - m[list(perm)], axis=1) / diag
        key = (float(errs.mean()), float(errs.max()), perm)
        if best is None or key < best:
            best = key
    assert best is not None
    return tuple(best[2]), best[0], best[1]


def semantic_to_manual(field_to_semantic: Iterable[int],
                       field_to_manual: Iterable[int]) -> tuple[int, ...]:
    out = [-1] * 4
    for sem, man in zip(field_to_semantic, field_to_manual):
        if out[int(sem)] != -1:
            raise ValueError("non-bijective calibration mapping")
        out[int(sem)] = int(man)
    if sorted(out) != [0, 1, 2, 3]:
        raise ValueError("non-bijective derived ordering")
    return tuple(out)
