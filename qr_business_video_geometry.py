#!/usr/bin/env python3
"""Map internally padded QR detections into original video coordinates."""
from __future__ import print_function

import copy
import math


ORIGINAL_POINT_FIELDS = (
    "ordered_corners",
    "stage1_geometry_corners",
    "stage2_crop_quad_source_xy",
    "coarse_ordered_corners",
)


def _translate_points(points, left, top, width, height):
    translated = []
    for point in points:
        if len(point) != 2:
            raise ValueError("each corner must contain exactly two coordinates")
        x = float(point[0]) - float(left)
        y = float(point[1]) - float(top)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("corner coordinates must be finite")
        translated.append([
            min(max(x, 0.0), float(width - 1)),
            min(max(y, 0.0), float(height - 1)),
        ])
    return translated


def _polygon_area(points):
    return 0.5 * abs(sum(
        point[0] * points[(index + 1) % len(points)][1]
        - point[1] * points[(index + 1) % len(points)][0]
        for index, point in enumerate(points)))


def original_frame_detection(detection, padding, width, height):
    """Return a copied detection in source pixels, or None for padding-only QR.

    ``stage2_roi_corners`` deliberately remain in the 112x112 ROI coordinate
    system. Every full-frame point field and the derived bounding box are
    translated into the original video's pixel coordinates.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("original frame dimensions must be positive")
    if len(padding) != 4:
        raise ValueError("padding must contain left, top, right and bottom")
    left, top, _, _ = padding
    if "ordered_corners" not in detection:
        raise ValueError("detection has no ordered corners")
    result = copy.deepcopy(detection)
    corners = _translate_points(
        detection["ordered_corners"], left, top, width, height)
    if len(corners) != 4:
        raise ValueError("a QR detection must contain exactly four corners")
    if _polygon_area(corners) < 1.0:
        return None
    result["ordered_corners"] = corners
    for field in ORIGINAL_POINT_FIELDS[1:]:
        if field in detection:
            result[field] = _translate_points(
                detection[field], left, top, width, height)
    result["derived_bbox_xyxy"] = [
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    ]
    result["coordinate_space"] = "original_video_pixels"
    if "stage2_roi_corners" in result:
        result["stage2_roi_coordinate_space"] = "stage2_112x112_pixels"
    return result


def original_frame_detections(detections, padding, width, height):
    """Convert valid detections and count detections entirely in padding."""
    converted = []
    rejected_padding_only = 0
    for detection in detections:
        result = original_frame_detection(detection, padding, width, height)
        if result is None:
            rejected_padding_only += 1
        else:
            converted.append(result)
    return converted, rejected_padding_only
