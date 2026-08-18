#!/usr/bin/env python3
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_dataset import QRImageTransform, bgr_to_yuv_tensor, instances_from_row
from qr_schema import CORNER_ORDER, SCHEMA_VERSION


def canonical(instances):
    return {"schema_version": SCHEMA_VERSION, "image": "images/a.jpg",
            "width": 240, "height": 320,
            "num_qrcodes": len(instances), "instances": instances}


def test_instances_schema_is_strict_and_multi_qr():
    quad = [[10, 10], [30, 10], [30, 30], [10, 30]]
    multi = instances_from_row(canonical([
        {"class_id": 0, "label": "qrcode", "corners": quad,
         "corner_order": CORNER_ORDER},
        {"class_id": 0, "label": "qrcode",
         "corners": (np.asarray(quad) + 40).tolist(),
         "corner_order": CORNER_ORDER}]))
    assert multi.shape == (2, 4, 2)
    assert instances_from_row(canonical([])).shape == (0, 4, 2)
    try:
        instances_from_row({"corners": quad})
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("legacy annotation must fail closed")


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
    test_instances_schema_is_strict_and_multi_qr()
    test_yuv_transform_has_three_channels()
    print("PASS")
