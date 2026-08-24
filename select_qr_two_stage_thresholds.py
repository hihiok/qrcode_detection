#!/usr/bin/env python3
"""Freeze joint stage-1/stage-2 thresholds using validation data only."""
from __future__ import print_function

import argparse
import hashlib
import json
import os

from eval_fsd_qr import evaluate_detector
from infer_qr_two_stage import TwoStageQRDetector


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(value):
    result = sorted(set(float(item) for item in value.split(",") if item.strip()))
    if not result or result[0] <= 0.0 or result[-1] >= 1.0:
        raise ValueError("thresholds must be within (0,1)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--stage2-thresholds", default="0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections-validation", type=int, default=20)
    parser.add_argument("--max-detections-final", type=int, default=1)
    parser.add_argument("--crop-margin", type=float, default=0.20)
    parser.add_argument("--stage2-min-geometry-iou", type=float, default=0.20)
    parser.add_argument("--stage1-opencv-refine", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage1_values = parse_thresholds(args.stage1_thresholds)
    stage2_values = parse_thresholds(args.stage2_thresholds)
    detector = TwoStageQRDetector(
        args.fsd_repo, args.stage1_checkpoint, args.stage2_checkpoint,
        args.device, min(stage1_values), min(stage2_values), args.nms_threshold,
        args.max_detections_validation, args.crop_margin,
        args.stage2_min_geometry_iou, args.stage1_opencv_refine)
    trials = []
    for stage1_threshold in stage1_values:
        detector.stage1.score_threshold = stage1_threshold
        for stage2_threshold in stage2_values:
            detector.stage2.score_threshold = stage2_threshold
            result = evaluate_detector(
                detector, args.validation_root, "val", args.match_iou_threshold)
            metrics = result["metrics"]
            trial = {"stage1_score_threshold": stage1_threshold,
                     "stage2_score_threshold": stage2_threshold,
                     "metrics": metrics}
            trials.append(trial)
            print(json.dumps(trial, sort_keys=True))
    # F1 first, then recall, then lower semantic corner error, then stricter
    # thresholds.  No final/test video result participates in selection.
    selected = max(trials, key=lambda item: (
        item["metrics"]["f1"], item["metrics"]["recall"],
        -item["metrics"]["mean_ordered_corner_error_px_matched"],
        item["stage1_score_threshold"], item["stage2_score_threshold"]))
    frozen = {
        "schema_version": "qr_two_stage_frozen_inference_v1",
        "selection_split": "validation_only",
        "validation_root": os.path.realpath(args.validation_root),
        "strict_eval_videos_used": False,
        "stage1_checkpoint": os.path.realpath(args.stage1_checkpoint),
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
        "stage2_checkpoint": os.path.realpath(args.stage2_checkpoint),
        "stage2_checkpoint_sha256": sha256_file(args.stage2_checkpoint),
        "stage1_score_threshold": selected["stage1_score_threshold"],
        "stage2_score_threshold": selected["stage2_score_threshold"],
        "nms_threshold": args.nms_threshold,
        "crop_margin": args.crop_margin,
        "stage2_min_geometry_iou": args.stage2_min_geometry_iou,
        "max_detections_validation": args.max_detections_validation,
        "max_detections_final": args.max_detections_final,
        "stage1_opencv_refine": bool(args.stage1_opencv_refine),
        "selected_validation_metrics": selected["metrics"],
        "trials": trials,
    }
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
    print(json.dumps(frozen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
