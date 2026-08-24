#!/usr/bin/env python3
"""Strict semantic-corner evaluation of the complete two-stage pipeline."""
from __future__ import print_function

import argparse
import json

from eval_fsd_qr import evaluate_detector
from infer_qr_two_stage import TwoStageQRDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-score-threshold", type=float, default=0.50)
    parser.add_argument("--stage2-score-threshold", type=float, default=0.70)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--crop-margin", type=float, default=0.20)
    parser.add_argument("--stage2-min-geometry-iou", type=float, default=0.20)
    parser.add_argument("--stage1-opencv-refine", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    detector = TwoStageQRDetector(
        args.fsd_repo, args.stage1_checkpoint, args.stage2_checkpoint,
        args.device, args.stage1_score_threshold, args.stage2_score_threshold,
        args.nms_threshold, args.max_detections, args.crop_margin,
        args.stage2_min_geometry_iou, args.stage1_opencv_refine)
    result = evaluate_detector(
        detector, args.data_root, args.split, args.match_iou_threshold)
    result["metrics"]["pipeline"] = "two_identical_fsd_structures"
    result["metrics"]["stage1_opencv_refine"] = bool(args.stage1_opencv_refine)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
