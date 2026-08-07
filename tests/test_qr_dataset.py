#!/usr/bin/env python3
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_dataset import QRImageTransform, bgr_to_yuv_tensor, instances_from_row


def test_instances_schema_and_legacy():
    quad = [[10, 10], [30, 10], [30, 30], [10, 30]]
    multi = instances_from_row({"instances": [
        {"corners": quad}, {"corners": np.asarray(quad) + 40}]})
    assert multi.shape == (2, 4, 2)
    assert instances_from_row({"corners": quad}).shape == (1, 4, 2)
    assert instances_from_row({"instances": []}).shape == (0, 4, 2)


def test_yuv_transform_has_three_channels():
    image = np.zeros((320, 240, 3), np.uint8)
    image[:, :, 2] = 255
    tensor = bgr_to_yuv_tensor(image)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    assert tensor.shape == (3, 320, 240)
    assert np.allclose(tensor.numpy().transpose(1, 2, 0),
                       expected.astype(np.float32) / 255.0)
    transformed, corners = QRImageTransform(False)(
        image, np.empty((0, 4, 2), np.float32))
    assert transformed.shape == (3, 320, 240)
    assert corners.shape == (0, 4, 2)


if __name__ == "__main__":
    test_instances_schema_and_legacy()
    test_yuv_transform_has_three_channels()
    print("PASS")
