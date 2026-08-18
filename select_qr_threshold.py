#!/usr/bin/env python3
"""Select score threshold on frozen public validation splits only."""
from __future__ import print_function

import argparse
import hashlib
import json
import os

import cv2

from dataset_v2_manifest import load_and_verify_manifest
from eval_fsd_qr import match_instances
from infer_fsd_qr import QRDetector
from qr_dataset import instances_from_row
from qr_schema import read_jsonl


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_grid(value):
    result = sorted(set(float(item) for item in value.split(",") if item.strip()))
    if not result or result[0] <= 0.0 or result[-1] >= 1.0:
        raise ValueError("Threshold grid must stay inside (0,1)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections-validation", type=int, default=20)
    parser.add_argument("--max-detections-final", type=int, default=1)
    parser.add_argument("--opencv-refine-final", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_and_verify_manifest(args.dataset_manifest)
    thresholds = parse_grid(args.thresholds)
    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, 240,
        min(thresholds), args.nms_threshold, 1000,
        args.max_detections_validation)
    counts = dict((threshold, {}) for threshold in thresholds)
    for source in manifest["sources"]:
        split_root = os.path.join(source["root"], "val")
        rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))
        per_threshold = dict((threshold, {"gt": 0, "pred": 0, "tp": 0,
                                                   "images": 0,
                                                   "negative_images": 0,
                                                   "negative_images_with_fp": 0})
                             for threshold in thresholds)
        for row in rows:
            image = cv2.imread(os.path.join(split_root, row["image"]),
                               cv2.IMREAD_COLOR)
            if image is None:
                raise IOError("Cannot read %s" % row["image"])
            gt = instances_from_row(row, row["image"])
            predictions = detector.predict(image)
            for threshold in thresholds:
                selected = [item for item in predictions
                            if item["score"] >= threshold]
                matched = match_instances(gt, selected,
                                          args.match_iou_threshold)
                per_threshold[threshold]["gt"] += len(gt)
                per_threshold[threshold]["pred"] += len(selected)
                per_threshold[threshold]["tp"] += len(matched)
                per_threshold[threshold]["images"] += 1
                if len(gt) == 0:
                    per_threshold[threshold]["negative_images"] += 1
                    per_threshold[threshold]["negative_images_with_fp"] += \
                        int(bool(selected))
        for threshold in thresholds:
            item = per_threshold[threshold]
            precision = item["tp"] / float(max(item["pred"], 1))
            recall = item["tp"] / float(max(item["gt"], 1))
            negative_fpr = item["negative_images_with_fp"] / float(
                max(item["negative_images"], 1))
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
            if item["gt"] == 0:
                selection_score = 1.0 - negative_fpr
            elif item["negative_images"]:
                selection_score = 0.8 * f1 + 0.2 * (1.0 - negative_fpr)
            else:
                selection_score = f1
            item.update({
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "negative_image_false_positive_rate": negative_fpr,
                "selection_score": selection_score,
            })
            counts[threshold][source["name"]] = item
    summary = []
    for threshold in thresholds:
        source_metrics = counts[threshold]
        macro_f1 = sum(item["f1"] for item in source_metrics.values()) / \
            float(len(source_metrics))
        macro_selection = sum(item["selection_score"]
                              for item in source_metrics.values()) / \
            float(len(source_metrics))
        summary.append({"threshold": threshold, "macro_source_f1": macro_f1,
                        "macro_source_selection_score": macro_selection,
                        "sources": source_metrics})
    # Higher threshold wins exact ties to reduce false positives.
    selected = max(summary, key=lambda item: (
                                              item["macro_source_selection_score"],
                                              item["threshold"]))
    output = {
        "schema_version": "qr_frozen_inference_config_v1",
        "selection_data": "dataset manifest validation splits only",
        "strict_eval_videos_used": False,
        "dataset_manifest": os.path.realpath(args.dataset_manifest),
        "checkpoint": os.path.realpath(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "score_threshold": selected["threshold"],
        "nms_threshold": args.nms_threshold,
        "max_detections_validation": args.max_detections_validation,
        "max_detections_final": args.max_detections_final,
        "opencv_refine_final": bool(args.opencv_refine_final),
        "refine_roi_expand": 0.18,
        "refine_min_iou": 0.20,
        "refine_max_shift": 0.40,
        "grid_results": summary,
    }
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
