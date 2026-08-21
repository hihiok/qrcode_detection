#!/usr/bin/env python3
"""Run two-stage ordered-corner QR inference on a video."""
from __future__ import print_function

import argparse
import json
import os
import time

import cv2

from infer_fsd_qr import draw_results
from infer_qr_two_stage import TwoStageQRDetector
from infer_video import create_writer, pad_frame_to_portrait_3x4
from strict_eval_guard import verify_final_input
from video_padding import compute_center_padding, padded_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-score-threshold", type=float, default=0.50)
    parser.add_argument("--stage2-score-threshold", type=float, default=0.70)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=1)
    parser.add_argument("--crop-margin", type=float, default=0.20)
    parser.add_argument("--stage2-min-geometry-iou", type=float, default=0.20)
    parser.add_argument("--stage1-opencv-refine", action="store_true")
    parser.add_argument("--pad-to-portrait-3x4", action="store_true")
    parser.add_argument("--pad-value", type=int, default=127)
    parser.add_argument("--strict-eval-manifest")
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("input and output paths must differ")
    strict_item = None
    if args.strict_eval_manifest:
        strict_item = verify_final_input(args.strict_eval_manifest, args.input)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise IOError("cannot open %s" % args.input)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.pad_to_portrait_3x4:
        initial_padding = compute_center_padding(width, height, 3, 4, True)
        output_size = padded_size(width, height, initial_padding)
    else:
        initial_padding = (0, 0, 0, 0)
        output_size = (width, height)
    detector = TwoStageQRDetector(
        args.fsd_repo, args.stage1_checkpoint, args.stage2_checkpoint,
        args.device, args.stage1_score_threshold, args.stage2_score_threshold,
        args.nms_threshold, args.max_detections, args.crop_margin,
        args.stage2_min_geometry_iou, args.stage1_opencv_refine)
    writer = create_writer(args.output, args.codec, fps, output_size)
    jsonl_path = os.path.splitext(args.output)[0] + ".jsonl"
    records = open(jsonl_path, "w")
    frame_index = total_stage1 = total_ordered = rejected_count = 0
    frames_with_ordered = 0
    started = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            padding = (0, 0, 0, 0)
            if args.pad_to_portrait_3x4:
                frame, padding = pad_frame_to_portrait_3x4(frame, args.pad_value)
            detections, rejected, stage1 = detector.predict(frame, True)
            total_stage1 += len(stage1)
            total_ordered += len(detections)
            rejected_count += len(rejected)
            frames_with_ordered += int(bool(detections))
            writer.write(draw_results(frame, detections))
            records.write(json.dumps({
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "padding": {"left": padding[0], "top": padding[1],
                            "right": padding[2], "bottom": padding[3]},
                "stage1_count": len(stage1),
                "ordered_detections": detections,
                "rejected": rejected,
            }) + "\n")
            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print("frames=%d fps=%.2f stage1=%d ordered=%d" %
                      (frame_index, frame_index / elapsed,
                       total_stage1, total_ordered))
    finally:
        capture.release()
        writer.release()
        records.close()
    elapsed = max(time.time() - started, 1e-6)
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump({
            "input": args.input,
            "output": args.output,
            "frames": frame_index,
            "elapsed_seconds": elapsed,
            "processing_fps": frame_index / elapsed,
            "stage1_detections": total_stage1,
            "ordered_detections": total_ordered,
            "stage2_rejections": rejected_count,
            "frames_with_ordered_detection": frames_with_ordered,
            "strict_final_evaluation": strict_item is not None,
            "strict_eval_video_sha256": strict_item["sha256"] if strict_item else None,
            "source_size": [width, height],
            "output_size": list(output_size),
            "padding": {"left": initial_padding[0], "top": initial_padding[1],
                        "right": initial_padding[2], "bottom": initial_padding[3]},
            "jsonl": jsonl_path,
        }, handle, indent=2)
    print("Wrote %s, %s and %s" % (args.output, jsonl_path, summary_path))


if __name__ == "__main__":
    main()
