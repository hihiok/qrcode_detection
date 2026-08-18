#!/usr/bin/env python3
"""Classical OpenCV ROI refinement for coarse neural QR quadrilaterals."""
from __future__ import print_function

import cv2
import numpy as np

from qr_common import corners_to_bbox, polygon_signed_area


def bbox_iou(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    left_top = np.maximum(a[:2], b[:2])
    right_bottom = np.minimum(a[2:], b[2:])
    size = np.maximum(0.0, right_bottom - left_top)
    intersection = float(size[0] * size[1])
    area_a = float(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]))
    area_b = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def valid_quad(points, width, height):
    points = np.asarray(points, np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        return False
    if ((points[:, 0] < 0).any() or (points[:, 0] >= width).any() or
            (points[:, 1] < 0).any() or (points[:, 1] >= height).any()):
        return False
    contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    return cv2.isContourConvex(contour) and polygon_signed_area(points) > 1.0


def align_to_semantic_order(candidate, coarse):
    """Use the network's correct P0..P3 identity to order refined pixels."""
    candidate = np.asarray(candidate, np.float32).reshape(4, 2)
    coarse = np.asarray(coarse, np.float32).reshape(4, 2)
    variants = []
    for base in (candidate, candidate[::-1]):
        for shift in range(4):
            value = np.roll(base, shift, axis=0)
            if polygon_signed_area(value) > 0:
                error = float(np.sqrt(((value - coarse) ** 2).sum(axis=1)).mean())
                variants.append((error, value.copy()))
    if not variants:
        return None, float("inf")
    return min(variants, key=lambda item: item[0])[1], \
        min(variants, key=lambda item: item[0])[0]


class OpenCVQRRefiner(object):
    def __init__(self, roi_expand=0.18, min_bbox_iou=0.20,
                 max_normalized_shift=0.40):
        self.detector = cv2.QRCodeDetector()
        self.roi_expand = float(roi_expand)
        self.min_bbox_iou = float(min_bbox_iou)
        self.max_normalized_shift = float(max_normalized_shift)

    def refine_one(self, image, detection):
        height, width = image.shape[:2]
        coarse = np.asarray(detection["ordered_corners"], np.float32).reshape(4, 2)
        box = corners_to_bbox(coarse, width, height)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        x1 = max(0, int(np.floor(box[0] - self.roi_expand * box_width)))
        y1 = max(0, int(np.floor(box[1] - self.roi_expand * box_height)))
        x2 = min(width, int(np.ceil(box[2] + self.roi_expand * box_width)) + 1)
        y2 = min(height, int(np.ceil(box[3] + self.roi_expand * box_height)) + 1)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return detection
        roi = image[y1:y2, x1:x2]
        try:
            found, points = self.detector.detect(roi)
        except cv2.error:
            return detection
        if not found or points is None:
            return detection
        candidate = np.asarray(points, np.float32).reshape(-1, 4, 2)[0]
        candidate[:, 0] += x1
        candidate[:, 1] += y1
        aligned, shift = align_to_semantic_order(candidate, coarse)
        if aligned is None or not valid_quad(aligned, width, height):
            return detection
        refined_box = corners_to_bbox(aligned, width, height)
        overlap = bbox_iou(box, refined_box)
        diagonal = float(np.sqrt(max(1e-9, box_width ** 2 + box_height ** 2)))
        normalized_shift = shift / diagonal
        if overlap < self.min_bbox_iou or normalized_shift > self.max_normalized_shift:
            return detection
        result = dict(detection)
        result["coarse_ordered_corners"] = detection["ordered_corners"]
        result["ordered_corners"] = [[float(x), float(y)] for x, y in aligned]
        result["derived_bbox_xyxy"] = [float(value) for value in refined_box]
        result["refinement"] = {
            "method": "opencv_qrcode_roi",
            "coarse_refined_bbox_iou": overlap,
            "mean_shift_over_coarse_diagonal": normalized_shift,
        }
        return result

    def refine(self, image, detections):
        return [self.refine_one(image, detection) for detection in detections]
