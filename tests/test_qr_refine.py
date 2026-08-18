#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_refine import align_to_semantic_order, bbox_iou


def test_refined_geometry_uses_network_semantic_identity():
    coarse = np.asarray([[80, 20], [82, 80], [20, 82], [18, 22]], np.float32)
    # Same physical points, OpenCV-like order beginning at another corner.
    candidate = np.asarray([[20, 80], [20, 20], [80, 20], [80, 80]], np.float32)
    aligned, error = align_to_semantic_order(candidate, coarse)
    expected = np.asarray([[80, 20], [80, 80], [20, 80], [20, 20]], np.float32)
    assert np.allclose(aligned, expected)
    assert error < 3.0


def test_bbox_iou_identity():
    assert abs(bbox_iou([1, 2, 10, 12], [1, 2, 10, 12]) - 1.0) < 1e-9


if __name__ == "__main__":
    test_refined_geometry_uses_network_semantic_identity()
    test_bbox_iou_identity()
    print("PASS")
