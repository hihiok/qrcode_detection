#!/usr/bin/env python3
"""Pre-generated 112x112 single-QR ROI dataset for semantic corner order."""
from __future__ import print_function

import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from qr_common import SEMANTIC_CORNER_ORDER, match_qr_instances, validate_semantic_corners
from qr_dataset import bgr_to_yuv_tensor


STAGE2_SIZE = 112
STAGE2_SCHEMA_VERSION = "qr_stage2_ordered_corners_112_v1"
STAGE2_MIN_BOXES = ((8, 12, 16), (24, 32, 40),
                    (56, 72), (88, 104, 112))


def read_rows(path):
    rows = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc))
            rows.append(validate_stage2_row(row, "%s:%d" % (path, line_number)))
    return rows


def validate_stage2_row(row, name="stage2 annotation"):
    if row.get("schema_version") != STAGE2_SCHEMA_VERSION:
        raise ValueError("%s has unsupported schema" % name)
    if (int(row.get("width", -1)), int(row.get("height", -1))) != \
            (STAGE2_SIZE, STAGE2_SIZE):
        raise ValueError("%s must be 112x112" % name)
    image = row.get("image")
    if not isinstance(image, str) or not image or os.path.isabs(image) or \
            os.path.normpath(image).startswith(".."):
        raise ValueError("%s has unsafe image path" % name)
    instances = row.get("instances")
    if not isinstance(instances, list) or len(instances) > 1:
        raise ValueError("%s must contain zero or one QR" % name)
    if int(row.get("num_qrcodes", -1)) != len(instances):
        raise ValueError("%s num_qrcodes mismatch" % name)
    normalized = []
    for instance in instances:
        if (instance.get("class_id") != 0 or instance.get("label") != "qrcode" or
                instance.get("corner_order") != list(SEMANTIC_CORNER_ORDER)):
            raise ValueError("%s has invalid QR identity/order" % name)
        corners = validate_semantic_corners(instance.get("corners"), name)
        if ((corners < 0.0).any() or (corners[:, 0] >= STAGE2_SIZE).any() or
                (corners[:, 1] >= STAGE2_SIZE).any()):
            raise ValueError("%s corners leave 112x112" % name)
        normalized.append(corners)
    result = dict(row)
    result["_validated_corners"] = normalized
    return result


class Stage2PhotometricTransform(object):
    """No crop/warp here: all geometry is pre-generated and auditable."""
    def __init__(self, training, seed=1234):
        self.training = bool(training)
        self.rng = np.random.RandomState(seed)

    def __call__(self, image):
        if image.shape[:2] != (STAGE2_SIZE, STAGE2_SIZE):
            raise ValueError("stage2 image must be 112x112")
        if not self.training:
            return bgr_to_yuv_tensor(image)
        value = image.astype(np.float32)
        value = np.clip(value * self.rng.uniform(0.75, 1.25) +
                        self.rng.uniform(-20.0, 20.0), 0.0, 255.0)
        if self.rng.rand() < 0.25:
            value += self.rng.normal(0.0, self.rng.uniform(1.0, 5.0), value.shape)
        value = np.clip(value, 0.0, 255.0).astype(np.uint8)
        if self.rng.rand() < 0.20:
            value = cv2.GaussianBlur(value, (3, 3), self.rng.uniform(0.2, 0.8))
        if self.rng.rand() < 0.20:
            quality = int(self.rng.randint(45, 92))
            ok, encoded = cv2.imencode(
                ".jpg", value, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                value = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return bgr_to_yuv_tensor(value)


class QRStage2Dataset(Dataset):
    def __init__(self, split_root, priors, training=False,
                 iou_threshold=0.35, seed=1234):
        self.split_root = os.path.abspath(split_root)
        self.rows = read_rows(os.path.join(self.split_root, "annotations.jsonl"))
        if not self.rows:
            raise RuntimeError("empty stage2 dataset: %s" % split_root)
        self.priors = priors.detach().cpu()
        self.iou_threshold = float(iou_threshold)
        self.transform = Stage2PhotometricTransform(training, seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = os.path.join(self.split_root, row["image"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("cannot read %s" % path)
        tensor = self.transform(image)
        values = row["_validated_corners"]
        corners = (np.stack(values).astype(np.float32) if values
                   else np.empty((0, 4, 2), np.float32))
        corners /= float(STAGE2_SIZE)
        labels, targets, _ = match_qr_instances(
            corners, self.priors, self.iou_threshold)
        return tensor, labels, targets
