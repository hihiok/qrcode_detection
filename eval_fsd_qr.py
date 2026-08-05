#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import math
import os

import cv2
import numpy as np

from infer_fsd_qr import QRDetector
from qr_common import corners_to_bbox, sort_polygon_for_geometry, validate_semantic_corners
from qr_dataset import read_jsonl


def bbox_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    tl, br = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    wh = np.maximum(0.0, br - tl)
    intersection = float(wh[0] * wh[1])
    area_a = float(max(0, a[2] - a[0]) * max(0, a[3] - a[1]))
    area_b = float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def polygon_iou(a, b):
    # Sorting is only for overlap geometry. Corner-error below preserves identity.
    a = sort_polygon_for_geometry(a)
    b = sort_polygon_for_geometry(b)
    area_a, area_b = abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b)))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return float(intersection) / max(area_a + area_b - float(intersection), 1e-9)


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", choices=("y", "rgb", "yuv444"), default="y")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="qr_eval.json")
    args = parser.parse_args()
    detector = QRDetector(args.fsd_repo, args.checkpoint, args.device,
                          args.input_mode, 240, args.score_threshold, 0.3, 400)
    split_root = os.path.join(args.data_root, args.split)
    rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))
    bbox_ious, polygon_ious, corner_errors, normalized_errors = [], [], [], []
    p0_errors = []
    detected = 0
    per_image = []
    for index, row in enumerate(rows):
        image = cv2.imread(os.path.join(split_root, row["image"]), cv2.IMREAD_COLOR)
        prediction = detector.predict(image)
        item = {"image": row["image"], "detected": prediction is not None}
        if prediction is not None:
            detected += 1
            gt = validate_semantic_corners(row["corners"], row["image"])
            pred = np.asarray(prediction["ordered_corners"], np.float32).reshape(4, 2)
            # No cyclic rematching: P0 must be correct, so direction errors are penalized.
            errors = np.sqrt(np.sum((gt - pred) ** 2, axis=1))
            gt_box = corners_to_bbox(gt)
            diagonal = math.sqrt((gt_box[2] - gt_box[0]) ** 2 +
                                 (gt_box[3] - gt_box[1]) ** 2)
            biou = bbox_iou(gt_box, prediction["derived_bbox_xyxy"])
            piou = polygon_iou(gt, pred)
            error = float(errors.mean())
            nme = error / max(diagonal, 1e-6)
            bbox_ious.append(biou)
            polygon_ious.append(piou)
            corner_errors.append(error)
            normalized_errors.append(nme)
            p0_errors.append(float(errors[0]))
            item.update({"score": prediction["score"], "derived_bbox_iou": biou,
                         "polygon_iou": piou, "p0_error_px": float(errors[0]),
                         "mean_ordered_corner_error_px": error,
                         "normalized_ordered_corner_error": nme})
        per_image.append(item)
        if (index + 1) % 500 == 0:
            print("%d/%d" % (index + 1, len(rows)))
    denominator = float(max(len(rows), 1))
    metrics = {
        "num_images": len(rows), "detection_rate": detected / denominator,
        "mean_derived_bbox_iou_detected": mean(bbox_ious),
        "mean_polygon_iou_detected": mean(polygon_ious),
        "mean_p0_error_px_detected": mean(p0_errors),
        "mean_ordered_corner_error_px_detected": mean(corner_errors),
        "mean_normalized_ordered_corner_error_detected": mean(normalized_errors),
        "ordered_corner_success_5px_all": sum(x <= 5 for x in corner_errors) / denominator,
        "ordered_corner_success_10px_all": sum(x <= 10 for x in corner_errors) / denominator}
    with open(args.output, "w") as handle:
        json.dump({"metrics": metrics, "per_image": per_image}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
