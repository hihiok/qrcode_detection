#!/usr/bin/env python3
"""Multi-QR dataset with semantic ordered-corner targets and YUV input."""
from __future__ import print_function

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, match_qr_instances,
                       validate_semantic_corners)


def read_jsonl(path):
    rows = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc))
    return rows


def instances_from_row(row, name="annotation"):
    """Read the canonical instances[] schema and legacy corners schemas."""
    if "instances" in row:
        raw = row["instances"]
        if not isinstance(raw, list):
            raise ValueError("%s instances must be a list" % name)
        values = [item["corners"] if isinstance(item, dict) else item for item in raw]
    elif "corners" in row:
        array = np.asarray(row["corners"], dtype=np.float32)
        if array.shape == (4, 2):
            values = [array]
        elif array.ndim == 3 and array.shape[1:] == (4, 2):
            values = list(array)
        else:
            raise ValueError("%s corners must be [4,2] or [M,4,2]" % name)
    else:
        values = []
    validated = [validate_semantic_corners(value, "%s instance %d" % (name, index))
                 for index, value in enumerate(values)]
    if not validated:
        return np.empty((0, 4, 2), dtype=np.float32)
    return np.stack(validated).astype(np.float32)


def transform_points_homography(points, matrix):
    shape = np.asarray(points).shape
    pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(shape)


def bgr_to_yuv_tensor(image):
    """OpenCV BGR -> YUV444, CHW float32 in [0,1]."""
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    tensor = torch.from_numpy(np.ascontiguousarray(converted.transpose(2, 0, 1)))
    return tensor.float().div_(255.0)


class QRImageTransform(object):
    def __init__(self, training, seed=1234):
        self.training = bool(training)
        self.rng = np.random.RandomState(seed)

    def _geometry(self, image, corners):
        # No horizontal mirror: it reverses semantic QR handedness.
        if corners.shape[0] == 0:
            return image, corners
        h, w = image.shape[:2]
        if self.rng.rand() < 0.40:
            margin = 0.025
            source = np.float32([[0, 0], [w - 1, 0],
                                 [w - 1, h - 1], [0, h - 1]])
            jitter = self.rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
            jitter[:, 0] *= w
            jitter[:, 1] *= h
            destination = source + jitter
            matrix = cv2.getPerspectiveTransform(source, destination)
            candidate = transform_points_homography(corners, matrix)
            in_frame = ((candidate[:, :, 0] >= 0).all() and
                        (candidate[:, :, 0] < w).all() and
                        (candidate[:, :, 1] >= 0).all() and
                        (candidate[:, :, 1] < h).all())
            if in_frame:
                try:
                    candidate = np.stack([
                        validate_semantic_corners(quad, "augmented instance %d" % index)
                        for index, quad in enumerate(candidate)])
                    image = cv2.warpPerspective(
                        image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
                    corners = candidate
                except ValueError:
                    pass
        return image, corners

    def _photometric(self, image):
        img = image.astype(np.float32)
        img = np.clip(img * self.rng.uniform(0.65, 1.35) +
                      self.rng.uniform(-28, 28), 0, 255)
        if self.rng.rand() < 0.30:
            img += self.rng.normal(0.0, self.rng.uniform(1.0, 8.0), size=img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.rng.rand() < 0.25:
            kernel = int(self.rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (kernel, kernel), self.rng.uniform(0.2, 1.3))
        if self.rng.rand() < 0.20:
            quality = int(self.rng.randint(35, 90))
            ok, encoded = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return img

    def __call__(self, image, corners):
        if image.shape[1] != INPUT_WIDTH or image.shape[0] != INPUT_HEIGHT:
            raise ValueError("Expected W,H=240,320; got %d,%d" %
                             (image.shape[1], image.shape[0]))
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2)
        if self.training:
            image, corners = self._geometry(image, corners)
            image = self._photometric(image)
        return bgr_to_yuv_tensor(image), corners


class QRDataset(Dataset):
    def __init__(self, split_root, priors, training=False,
                 iou_threshold=0.35, seed=1234):
        self.split_root = os.path.abspath(split_root)
        self.rows = read_jsonl(os.path.join(self.split_root, "annotations.jsonl"))
        self.priors = priors.detach().cpu()
        self.transform = QRImageTransform(training, seed)
        self.iou_threshold = float(iou_threshold)
        if not self.rows:
            raise RuntimeError("No annotations in %s" % self.split_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = os.path.join(self.split_root, row["image"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Cannot read %s" % path)
        corners = instances_from_row(row, path)
        image_tensor, corners = self.transform(image, corners)
        corners_norm = corners / np.asarray([INPUT_WIDTH, INPUT_HEIGHT], np.float32)
        labels, targets, _ = match_qr_instances(
            corners_norm, self.priors, self.iou_threshold)
        return image_tensor, labels, targets


# Compatibility alias for older imports; semantics are now zero-or-more QR.
SingleQRDataset = QRDataset
