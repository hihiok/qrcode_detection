#!/usr/bin/env python3
"""Shared geometry, prior, output and checkpoint helpers for FSD QR."""
from __future__ import print_function

import json
import math
import os
import sys

import cv2
import numpy as np
import torch


INPUT_WIDTH = 240
INPUT_HEIGHT = 320
NUM_CLASSES = 2
CENTER_VARIANCE = 0.1
STRIDES = (8, 16, 32, 64)
MIN_BOXES = ((10, 16, 22), (28, 36, 48), (64, 96), (128, 192, 256))
SEMANTIC_CORNER_ORDER = ("qr_top_left", "qr_top_right",
                         "qr_bottom_right", "qr_bottom_left")


def polygon_signed_area(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) -
                              np.roll(pts[:, 0], -1) * pts[:, 1]))


def validate_semantic_corners(points, name="corners", min_area=1.0):
    """Validate P0,P1,P2,P3 without changing their semantic identities.

    P0 is QR-native top-left, not the point nearest the image top-left.  The
    sequence must remain clockwise after rotation/perspective transforms.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("%s must have shape (4,2); got %s" % (name, pts.shape))
    if not np.isfinite(pts).all():
        raise ValueError("%s contains NaN or Inf" % name)
    area = polygon_signed_area(pts)
    if area <= float(min_area):
        raise ValueError(
            "%s must be P0=QR-TL,P1=QR-TR,P2=QR-BR,P3=QR-BL clockwise; "
            "signed area is %.6f" % (name, area))
    contour = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise ValueError("%s is self-crossing or non-convex" % name)
    return pts.copy()


def sort_polygon_for_geometry(points):
    """Sort a copy clockwise for geometry only; never use for target identity."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    if polygon_signed_area(pts) < 0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def corners_to_bbox(points, width=None, height=None):
    """Derive [xmin,ymin,xmax,ymax] from corners; this is not a model output."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    if width is not None:
        x1, x2 = np.clip([x1, x2], 0, width - 1)
    if height is not None:
        y1, y2 = np.clip([y1, y2], 0, height - 1)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def corners_tensor_to_boxes(corners):
    """Convert [...,8] semantic corners to derived [...,4] xyxy boxes."""
    points = corners.reshape(*corners.shape[:-1], 4, 2)
    minimum = points.min(dim=-2)[0]
    maximum = points.max(dim=-2)[0]
    return torch.cat([minimum, maximum], dim=-1)


def rotate_image_points_cw(image, points):
    """Rotate image 90 degrees clockwise; point identity is preserved."""
    old_h = image.shape[0]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    out = np.empty_like(pts)
    out[:, 0] = old_h - 1 - pts[:, 1]
    out[:, 1] = pts[:, 0]
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), out


def letterbox_image_points(image, points, width=INPUT_WIDTH,
                           height=INPUT_HEIGHT, pad_value=127):
    old_h, old_w = image.shape[:2]
    scale = min(float(width) / old_w, float(height) / old_h)
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = pts[:, 0] * scale + left
    pts[:, 1] = pts[:, 1] * scale + top
    return canvas, pts, {"scale": scale, "left": left, "top": top,
                         "source_width": old_w, "source_height": old_h}


def generate_portrait_priors(width=INPUT_WIDTH, height=INPUT_HEIGHT,
                             strides=STRIDES, min_boxes=MIN_BOXES):
    """Generate priors in the H,W,anchor flatten order used by SSD heads."""
    if strides is None:
        strides = STRIDES
    if min_boxes is None:
        min_boxes = MIN_BOXES
    priors = []
    feature_shapes = []
    for stride, layer_boxes in zip(strides, min_boxes):
        fmap_h = int(math.ceil(float(height) / stride))
        fmap_w = int(math.ceil(float(width) / stride))
        feature_shapes.append((fmap_h, fmap_w))
        for j in range(fmap_h):
            for i in range(fmap_w):
                cx = (i + 0.5) * stride / float(width)
                cy = (j + 0.5) * stride / float(height)
                for min_box in layer_boxes:
                    priors.append([cx, cy, min_box / float(width),
                                   min_box / float(height)])
    return torch.tensor(priors, dtype=torch.float32).clamp_(0.0, 1.0), feature_shapes


def center_to_corner(boxes):
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2], dim=1)


def box_iou(boxes0, boxes1, eps=1e-9):
    overlap_tl = torch.max(boxes0[:, None, :2], boxes1[None, :, :2])
    overlap_br = torch.min(boxes0[:, None, 2:], boxes1[None, :, 2:])
    overlap = torch.clamp(overlap_br - overlap_tl, min=0)
    intersection = overlap[:, :, 0] * overlap[:, :, 1]
    area0 = torch.clamp(boxes0[:, 2] - boxes0[:, 0], min=0) * \
        torch.clamp(boxes0[:, 3] - boxes0[:, 1], min=0)
    area1 = torch.clamp(boxes1[:, 2] - boxes1[:, 0], min=0) * \
        torch.clamp(boxes1[:, 3] - boxes1[:, 1], min=0)
    return intersection / (area0[:, None] + area1[None, :] - intersection + eps)


def _unique_forced_matches(ious):
    """Return unique prior indices, one for every GT, using greedy max IoU."""
    num_priors, num_gt = ious.shape
    if num_gt == 0:
        return torch.empty((0,), dtype=torch.long)
    forced = torch.full((num_gt,), -1, dtype=torch.long)
    used = torch.zeros((num_priors,), dtype=torch.bool)
    # Assign the hardest-to-place GT first (highest available IoU).  QR counts
    # are small and priors are numerous, so this produces stable unique pairs.
    remaining = list(range(num_gt))
    while remaining:
        choices = []
        for gt_index in remaining:
            values = ious[:, gt_index].clone()
            values[used] = -1.0
            score, prior_index = torch.max(values, dim=0)
            choices.append((float(score.item()), gt_index, int(prior_index.item())))
        _, gt_index, prior_index = max(choices, key=lambda item: item[0])
        forced[gt_index] = prior_index
        used[prior_index] = True
        remaining.remove(gt_index)
    return forced


def match_qr_instances(corners, priors, iou_threshold=0.35):
    """Assign every prior to at most one QR and encode its matched corners.

    Args:
        corners: [M,4,2] normalized semantic P0..P3. M may be zero.
        priors: [A,4] center-form SSD priors.
    Returns:
        labels [A], encoded_targets [A,8], matched_gt [A] (-1 for background).
    """
    points = torch.as_tensor(corners, dtype=torch.float32).reshape(-1, 4, 2)
    num_priors = priors.size(0)
    labels = torch.zeros((num_priors,), dtype=torch.long)
    targets = torch.zeros((num_priors, 8), dtype=torch.float32)
    matched_gt = torch.full((num_priors,), -1, dtype=torch.long)
    if points.size(0) == 0:
        return labels, targets, matched_gt

    gt_boxes = torch.cat([points.min(dim=1)[0], points.max(dim=1)[0]], dim=1)
    ious = box_iou(center_to_corner(priors), gt_boxes)
    best_iou, best_gt = torch.max(ious, dim=1)
    positive = best_iou >= float(iou_threshold)

    forced_priors = _unique_forced_matches(ious)
    for gt_index, prior_index in enumerate(forced_priors.tolist()):
        positive[prior_index] = True
        best_gt[prior_index] = gt_index

    labels[positive] = 1
    matched_gt[positive] = best_gt[positive]
    matched_points = points[best_gt.clamp(min=0)]
    targets = encode_ordered_corners(matched_points, priors)
    targets[~positive] = 0.0
    return labels, targets, matched_gt


def encode_ordered_corners(corners, priors):
    """Encode semantic P0..P3 relative to priors.

    ``corners`` may be one [4,2] quad (broadcast to every prior) or one
    matched [A,4,2] quad per prior.
    """
    points = torch.as_tensor(corners, dtype=torch.float32).reshape(-1, 4, 2)
    if points.size(0) == 1:
        points = points.expand(priors.size(0), 4, 2)
    if points.size(0) != priors.size(0):
        raise ValueError("corners/priors mismatch: %d vs %d" %
                         (points.size(0), priors.size(0)))
    encoded = (points - priors[:, None, :2]) / \
        (CENTER_VARIANCE * priors[:, None, 2:])
    return encoded.reshape(priors.size(0), 8)


def decode_ordered_corners(encoded, priors):
    points = encoded.reshape(-1, 4, 2)
    return points * (CENTER_VARIANCE * priors[:, None, 2:]) + priors[:, None, :2]


def hard_nms(boxes, scores, iou_threshold=0.3, top_k=200):
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < top_k:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(boxes[current:current + 1], boxes[rest]).squeeze(0)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def unpack_outputs(outputs, num_priors, num_classes=NUM_CLASSES):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("FSD must return tuple/list, got %s" % type(outputs))
    confidence = corners = None
    for value in outputs:
        if not torch.is_tensor(value) or value.dim() != 3 or value.size(1) != num_priors:
            continue
        if value.size(2) == num_classes:
            confidence = value
        elif value.size(2) == 8:
            corners = value
    if confidence is None or corners is None:
        shapes = [tuple(x.shape) for x in outputs if torch.is_tensor(x)]
        raise RuntimeError("Expected [N,%d,2] and [N,%d,8]; got %s" %
                           (num_priors, num_priors, shapes))
    return confidence, corners


def extract_state_dict(obj):
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("checkpoint is not a model/state_dict dictionary")
    state = {}
    for key, value in obj.items():
        state[key[7:] if key.startswith("module.") else key] = value
    return state


def is_regression_key(key):
    return "regression_headers" in key


def load_fd_pretrained(model, checkpoint_path):
    """Load FSD weights, intentionally reinitializing the 4->8 regression head."""
    loaded = extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    current = model.state_dict()
    usable = {}
    unexpected = []
    bad_shapes = []
    for key, value in loaded.items():
        if is_regression_key(key):
            continue
        if key not in current:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(current[key].shape):
            if (value.dim() == 4 and current[key].dim() == 4 and
                    value.size(1) == 1 and current[key].size(1) == 3 and
                    value.size(0) == current[key].size(0) and
                    tuple(value.shape[2:]) == tuple(current[key].shape[2:])):
                adapted = torch.zeros_like(current[key])
                adapted[:, 0:1] = value
                usable[key] = adapted
                print("Adapted single-Y input tensor to YUV: %s" % key)
            else:
                bad_shapes.append((key, tuple(value.shape), tuple(current[key].shape)))
        else:
            usable[key] = value
    if unexpected:
        raise RuntimeError("Unexpected non-regression keys: %s" % unexpected[:20])
    if bad_shapes:
        raise RuntimeError("Non-regression shape mismatches: %s" % bad_shapes[:20])
    missing = [key for key in current if key not in usable]
    illegal = [key for key in missing if not is_regression_key(key)]
    if illegal:
        raise RuntimeError("Missing non-regression weights: %s" % illegal[:20])
    result = model.load_state_dict(usable, strict=False)
    if list(result.unexpected_keys):
        raise RuntimeError("Unexpected keys after load: %s" % list(result.unexpected_keys))
    print("Loaded %d FSD tensors; initialized %d ordered-corner head tensors." %
          (len(usable), len(missing)))
    return missing


def load_qr_checkpoint_strict(model, checkpoint_path):
    loaded = extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    current = model.state_dict()
    adapted_keys = []
    for key, value in list(loaded.items()):
        if key not in current or tuple(value.shape) == tuple(current[key].shape):
            continue
        if (value.dim() == 4 and current[key].dim() == 4 and
                value.size(1) == 1 and current[key].size(1) == 3 and
                value.size(0) == current[key].size(0) and
                tuple(value.shape[2:]) == tuple(current[key].shape[2:])):
            adapted = torch.zeros_like(current[key])
            adapted[:, 0:1] = value
            loaded[key] = adapted
            adapted_keys.append(key)
    model.load_state_dict(loaded, strict=True)
    if adapted_keys:
        print("Adapted QR checkpoint single-Y input to YUV: %s" % adapted_keys)


def save_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def add_repo_to_path(repo_root):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root
