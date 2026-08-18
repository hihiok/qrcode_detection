#!/usr/bin/env python3
"""Evaluate multi-QR predictions with one-to-one instance matching."""
from __future__ import print_function

import argparse
import json
import math
import os

import cv2
import numpy as np

from infer_fsd_qr import QRDetector
from qr_common import corners_to_bbox, sort_polygon_for_geometry
from qr_dataset import instances_from_row, read_jsonl


def bbox_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    tl, br = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    wh = np.maximum(0.0, br - tl)
    intersection = float(wh[0] * wh[1])
    area_a = float(max(0, a[2] - a[0]) * max(0, a[3] - a[1]))
    area_b = float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def polygon_iou(a, b):
    a = sort_polygon_for_geometry(a)
    b = sort_polygon_for_geometry(b)
    area_a, area_b = abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b)))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return float(intersection) / max(area_a + area_b - float(intersection), 1e-9)


def match_instances(gt_corners, predictions, iou_threshold):
    candidates = []
    for gt_index, gt in enumerate(gt_corners):
        gt_box = corners_to_bbox(gt)
        for pred_index, prediction in enumerate(predictions):
            iou = bbox_iou(gt_box, prediction["derived_bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    matches = []
    used_gt, used_pred = set(), set()
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", choices=("yuv", "yuv444"), default="yuv")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--output", default="qr_eval.json")
    args = parser.parse_args()
    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, 240,
        args.score_threshold, args.nms_threshold, 400, args.max_detections)
    split_root = os.path.join(args.data_root, args.split)
    rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))

    bbox_ious, polygon_ious, corner_errors, normalized_errors, p0_errors = (
        [], [], [], [], [])
    total_gt = total_predictions = true_positives = 0
    negative_images = negative_images_with_fp = 0
    per_image = []
    for index, row in enumerate(rows):
        image = cv2.imread(os.path.join(split_root, row["image"]), cv2.IMREAD_COLOR)
        gt_corners = instances_from_row(row, row["image"])
        predictions = detector.predict(image)
        matches = match_instances(
            gt_corners, predictions, args.match_iou_threshold)
        total_gt += len(gt_corners)
        total_predictions += len(predictions)
        true_positives += len(matches)
        if len(gt_corners) == 0:
            negative_images += 1
            negative_images_with_fp += int(bool(predictions))
        item = {
            "image": row["image"], "num_gt": len(gt_corners),
            "num_predictions": len(predictions), "num_matches": len(matches),
            "matches": []}
        for gt_index, pred_index, biou in matches:
            gt = gt_corners[gt_index]
            prediction = predictions[pred_index]
            pred = np.asarray(prediction["ordered_corners"], np.float32).reshape(4, 2)
            # Strict semantic identity: no cyclic corner rematching.
            errors = np.sqrt(np.sum((gt - pred) ** 2, axis=1))
            gt_box = corners_to_bbox(gt)
            diagonal = math.sqrt((gt_box[2] - gt_box[0]) ** 2 +
                                 (gt_box[3] - gt_box[1]) ** 2)
            piou = polygon_iou(gt, pred)
            error = float(errors.mean())
            nme = error / max(diagonal, 1e-6)
            bbox_ious.append(biou)
            polygon_ious.append(piou)
            corner_errors.append(error)
            normalized_errors.append(nme)
            p0_errors.append(float(errors[0]))
            item["matches"].append({
                "gt_index": gt_index, "prediction_index": pred_index,
                "score": prediction["score"], "derived_bbox_iou": biou,
                "polygon_iou": piou, "p0_error_px": float(errors[0]),
                "mean_ordered_corner_error_px": error,
                "normalized_ordered_corner_error": nme})
        per_image.append(item)
        if (index + 1) % 500 == 0:
            print("%d/%d" % (index + 1, len(rows)))

    false_positives = total_predictions - true_positives
    false_negatives = total_gt - true_positives
    precision = true_positives / float(max(total_predictions, 1))
    recall = true_positives / float(max(total_gt, 1))
    metrics = {
        "num_images": len(rows), "num_gt_instances": total_gt,
        "num_predictions": total_predictions, "true_positives": true_positives,
        "false_positives": false_positives, "false_negatives": false_negatives,
        "precision": precision, "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
        "negative_images": negative_images,
        "negative_image_false_positive_rate":
            negative_images_with_fp / float(max(negative_images, 1)),
        "mean_derived_bbox_iou_matched": mean(bbox_ious),
        "mean_polygon_iou_matched": mean(polygon_ious),
        "mean_p0_error_px_matched": mean(p0_errors),
        "mean_ordered_corner_error_px_matched": mean(corner_errors),
        "mean_normalized_ordered_corner_error_matched": mean(normalized_errors),
        "ordered_corner_success_5px_matched":
            sum(x <= 5 for x in corner_errors) / float(max(len(corner_errors), 1)),
        "ordered_corner_success_10px_matched":
            sum(x <= 10 for x in corner_errors) / float(max(len(corner_errors), 1))}
    with open(args.output, "w") as handle:
        json.dump({"metrics": metrics, "per_image": per_image}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
