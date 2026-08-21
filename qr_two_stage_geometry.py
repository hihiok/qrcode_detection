#!/usr/bin/env python3
"""Geometry shared by the stage-2 crop builder and two-stage inference."""
from __future__ import print_function

import cv2
import numpy as np

from qr_common import INPUT_HEIGHT, INPUT_WIDTH, polygon_signed_area


def destination_quad(width=INPUT_WIDTH, height=INPUT_HEIGHT):
    return np.float32([
        [0.0, 0.0],
        [float(width) - 1.0, 0.0],
        [float(width) - 1.0, float(height) - 1.0],
        [0.0, float(height) - 1.0],
    ])


def order_quad_for_crop(points):
    """Return image-geometric TL,TR,BR,BL without using QR semantics."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all() or len(np.unique(pts, axis=0)) != 4:
        raise ValueError("crop quad must contain four distinct finite points")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    if polygon_signed_area(ordered) < 0.0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    contour = np.rint(ordered).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(polygon_signed_area(ordered)) < 1.0:
        raise ValueError("crop quad is degenerate or non-convex")
    return ordered.astype(np.float32)


def expand_quad(points, margin):
    ordered = order_quad_for_crop(points)
    factor = 1.0 + 2.0 * float(margin)
    if factor <= 0.0:
        raise ValueError("margin must be greater than -0.5")
    center = ordered.mean(axis=0)
    return (center + (ordered - center) * factor).astype(np.float32)


def jitter_quad(points, rng, fraction, max_attempts=30):
    """Jitter crop vertices while retaining a valid convex correspondence."""
    ordered = order_quad_for_crop(points)
    fraction = float(fraction)
    if fraction <= 0.0:
        return ordered
    edges = np.sqrt(np.sum((ordered - np.roll(ordered, -1, axis=0)) ** 2, axis=1))
    scale = max(float(np.median(edges)), 1.0) * fraction
    for _ in range(int(max_attempts)):
        candidate = ordered + rng.normal(0.0, scale, size=(4, 2)).astype(np.float32)
        try:
            candidate = order_quad_for_crop(candidate)
        except ValueError:
            continue
        # Reject a reordering that changes vertex correspondence.  A jittered
        # vertex must remain closest to the same original vertex.
        distances = np.sqrt(np.sum(
            (candidate[:, None, :] - ordered[None, :, :]) ** 2, axis=2))
        if np.array_equal(np.argmin(distances, axis=1), np.arange(4)):
            return candidate
    return ordered


def crop_homography(geometry_corners, margin=0.18, rng=None, jitter=0.0,
                    output_width=INPUT_WIDTH, output_height=INPUT_HEIGHT):
    crop_quad = expand_quad(geometry_corners, margin)
    if rng is not None and float(jitter) > 0.0:
        crop_quad = jitter_quad(crop_quad, rng, jitter)
    destination = destination_quad(output_width, output_height)
    matrix = cv2.getPerspectiveTransform(crop_quad, destination)
    inverse = cv2.getPerspectiveTransform(destination, crop_quad)
    return crop_quad, matrix, inverse


def transform_points(points, matrix):
    pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(matrix, np.float32))[0]


def warp_qr_roi(image, geometry_corners, margin=0.18, rng=None, jitter=0.0,
                border_value=127, output_width=INPUT_WIDTH,
                output_height=INPUT_HEIGHT):
    crop_quad, matrix, inverse = crop_homography(
        geometry_corners, margin, rng, jitter, output_width, output_height)
    roi = cv2.warpPerspective(
        image, matrix, (int(output_width), int(output_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(int(border_value),) * 3)
    return roi, crop_quad, matrix, inverse


def convex_iou(first, second):
    a = order_quad_for_crop(first)
    b = order_quad_for_crop(second)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return float(intersection) / max(union, 1e-9)
