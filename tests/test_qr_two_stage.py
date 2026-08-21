#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_qr_stage2_dataset import build_dataset, verify_dataset
from qr_common import SEMANTIC_CORNER_ORDER, generate_portrait_priors
from qr_schema import SCHEMA_VERSION
from qr_stage2_dataset import STAGE2_MIN_BOXES
from qr_two_stage_geometry import (convex_iou, order_quad_for_crop,
                                   transform_points, warp_qr_roi)
from strict_eval_guard import freeze_manifest


def test_geometry_crop_round_trip_preserves_semantic_identity():
    priors, feature_shapes = generate_portrait_priors(
        112, 112, min_boxes=STAGE2_MIN_BOXES)
    assert feature_shapes == [(14, 14), (7, 7), (4, 4), (2, 2)]
    assert priors.shape == (779, 4)
    image = np.zeros((320, 240, 3), np.uint8)
    # QR-native P0 is physically top-right after rotation.
    semantic = np.float32([[180, 60], [180, 250], [45, 250], [45, 60]])
    geometry = order_quad_for_crop(semantic)
    assert np.allclose(geometry[0], [45, 60])
    roi, _, matrix, inverse = warp_qr_roi(
        image, geometry, margin=0.20, output_width=112, output_height=112)
    assert roi.shape == (112, 112, 3)
    transformed = transform_points(semantic, matrix)
    restored = transform_points(transformed, inverse)
    assert np.allclose(restored, semantic, atol=1e-3)
    assert convex_iou(transformed, transform_points(geometry, matrix)) > 0.99


def _row(image_name, corners):
    instances = [] if corners is None else [{
        "class_id": 0, "label": "qrcode",
        "corners": np.asarray(corners, float).tolist(),
        "corner_order": list(SEMANTIC_CORNER_ORDER)}]
    return {"schema_version": SCHEMA_VERSION, "image": image_name,
            "width": 240, "height": 320,
            "num_qrcodes": len(instances), "instances": instances}


def test_stage2_dataset_build_and_verify():
    root = tempfile.mkdtemp(prefix="qr_stage2_test_")
    try:
        source = os.path.join(root, "source")
        corners = [[45, 60], [190, 65], [185, 255], [40, 250]]
        for split_index, split in enumerate(("train", "val", "test")):
            split_root = os.path.join(source, split)
            os.makedirs(os.path.join(split_root, "images"))
            positive = np.full((320, 240, 3), 80 + split_index, np.uint8)
            negative = np.full((320, 240, 3), 20 + split_index, np.uint8)
            cv2.imwrite(os.path.join(split_root, "images", "positive.jpg"), positive)
            cv2.imwrite(os.path.join(split_root, "images", "negative.jpg"), negative)
            with open(os.path.join(split_root, "annotations.jsonl"), "w") as handle:
                handle.write(json.dumps(_row("images/positive.jpg", corners)) + "\n")
                handle.write(json.dumps(_row("images/negative.jpg", None)) + "\n")
        video = os.path.join(root, "vrtest.mp4")
        with open(video, "wb") as handle:
            handle.write(b"strict-video")
        strict_manifest = os.path.join(root, "strict.json")
        freeze_manifest([video], strict_manifest)
        output = os.path.join(root, "stage2")
        args = argparse.Namespace(
            source=["tiny=%s" % source], dataset_manifest=None,
            strict_eval_manifest=strict_manifest, output=output,
            train_variants=2, val_variants=1, test_variants=1,
            negative_per_image=1, margin_min=0.12, margin_max=0.30,
            jitter=0.04, seed=20260821)
        manifest = build_dataset(args)
        assert manifest["frozen_stats"]["train"]["instances"] == 2
        assert manifest["frozen_stats"]["train"]["negative_images"] == 1
        assert verify_dataset(output)["pass"]
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_geometry_crop_round_trip_preserves_semantic_identity()
    test_stage2_dataset_build_and_verify()
    print("PASS")
