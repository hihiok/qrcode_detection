#!/usr/bin/env python3
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_common import (corners_tensor_to_boxes, decode_ordered_corners,
                       encode_ordered_corners, generate_portrait_priors,
                       rotate_image_points_cw, validate_semantic_corners)


def test_portrait_priors():
    priors, feature_shapes = generate_portrait_priors()
    assert feature_shapes == [(40, 30), (20, 15), (10, 8), (5, 4)]
    assert priors.shape == (4420, 4)


def test_rotation_preserves_identity_not_image_tl():
    image = np.zeros((240, 320, 3), np.uint8)
    original = np.float32([[20, 30], [200, 30], [200, 180], [20, 180]])
    rotated_image, rotated = rotate_image_points_cw(image, original)
    assert rotated_image.shape[:2] == (320, 240)
    # After clockwise rotation, QR-native P0 is physically top-right.
    assert np.allclose(rotated[0], [209, 20])
    assert np.allclose(rotated[1], [209, 200])
    assert np.allclose(validate_semantic_corners(rotated), rotated)


def test_corner_codec_and_derived_bbox():
    priors, _ = generate_portrait_priors()
    corners = np.float32([[0.15, 0.20], [0.75, 0.22],
                          [0.72, 0.80], [0.18, 0.78]])
    encoded = encode_ordered_corners(corners, priors)
    decoded = decode_ordered_corners(encoded, priors)
    expected = torch.from_numpy(corners).unsqueeze(0).expand(priors.size(0), 4, 2)
    assert torch.allclose(decoded, expected, atol=1e-5)
    boxes = corners_tensor_to_boxes(decoded.reshape(priors.size(0), 8))
    assert torch.allclose(boxes[0], torch.tensor([0.15, 0.20, 0.75, 0.80]), atol=1e-5)


if __name__ == "__main__":
    test_portrait_priors()
    test_rotation_preserves_identity_not_image_tl()
    test_corner_codec_and_derived_bbox()
    print("PASS")
