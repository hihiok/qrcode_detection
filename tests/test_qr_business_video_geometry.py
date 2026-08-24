#!/usr/bin/env python3
"""Dependency-free tests for original-resolution QR coordinate recovery."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_business_video_geometry import (original_frame_detection,
                                        original_frame_detections)
from video_padding import compute_center_padding


def detection(corners):
    return {
        "score": 0.92,
        "stage1_score": 0.96,
        "stage2_score": 0.92,
        "ordered_corners": corners,
        "stage1_geometry_corners": corners,
        "stage2_crop_quad_source_xy": corners,
        "stage2_roi_corners": [[8, 8], [103, 8], [103, 103], [8, 103]],
        "derived_bbox_xyxy": [0, 0, 0, 0],
    }


def test_center_padding_is_removed_without_rescaling():
    padding = compute_center_padding(480, 408, 3, 4, True)
    assert padding == (0, 116, 0, 116)
    padded = [[100, 166], [260, 166], [260, 326], [100, 326]]
    original = original_frame_detection(detection(padded), padding, 480, 408)
    expected = [[100.0, 50.0], [260.0, 50.0],
                [260.0, 210.0], [100.0, 210.0]]
    assert original["ordered_corners"] == expected
    assert original["stage1_geometry_corners"] == expected
    assert original["stage2_crop_quad_source_xy"] == expected
    assert original["derived_bbox_xyxy"] == [100.0, 50.0, 260.0, 210.0]
    assert original["stage2_roi_corners"][0] == [8, 8]
    assert original["coordinate_space"] == "original_video_pixels"


def test_horizontal_padding_is_removed_and_edges_are_clipped():
    padded = [[5, 12], [105, 12], [105, 102], [5, 102]]
    result = original_frame_detection(detection(padded), (10, 2, 10, 2), 90, 100)
    assert result["ordered_corners"] == [
        [0.0, 10.0], [89.0, 10.0], [89.0, 99.0], [0.0, 99.0]]


def test_padding_only_detections_are_rejected():
    padded_only = detection([[10, 10], [50, 10], [50, 50], [10, 50]])
    valid = detection([[100, 166], [260, 166], [260, 326], [100, 326]])
    converted, rejected = original_frame_detections(
        [padded_only, valid], (0, 116, 0, 116), 480, 408)
    assert len(converted) == 1
    assert rejected == 1


def test_zero_padding_preserves_original_coordinates():
    corners = [[5, 7], [80, 7], [80, 90], [5, 90]]
    result = original_frame_detection(detection(corners), (0, 0, 0, 0), 100, 120)
    assert result["ordered_corners"] == corners


if __name__ == "__main__":
    test_center_padding_is_removed_without_rescaling()
    test_horizontal_padding_is_removed_and_edges_are_clipped()
    test_padding_only_detections_are_rejected()
    test_zero_padding_preserves_original_coordinates()
    print("PASS: original-resolution video coordinate recovery")
