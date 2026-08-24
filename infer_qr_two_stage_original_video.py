#!/usr/bin/env python3
"""Run two-stage QR inference and preserve original video size/coordinates."""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import time

import cv2

from infer_qr_two_stage import TwoStageQRDetector
from infer_video import create_writer, pad_frame_to_portrait_3x4
from qr_business_video_geometry import original_frame_detections
from strict_eval_guard import verify_final_input
from video_padding import compute_center_padding


CSV_FIELDS = [
    "frame_index", "time_seconds", "detection_index", "score",
    "stage1_score", "stage2_score", "stage2_geometry_iou",
    "p0_x", "p0_y", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y",
]
CORNER_COLORS = ((0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True,
                        help="Annotated MP4 at exactly the input resolution")
    parser.add_argument("--jsonl-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--frozen-config",
                        help="Validation-frozen two-stage threshold JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-score-threshold", type=float)
    parser.add_argument("--stage2-score-threshold", type=float)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--crop-margin", type=float)
    parser.add_argument("--stage2-min-geometry-iou", type=float)
    parser.add_argument("--stage1-opencv-refine", action="store_true")
    parser.add_argument("--pad-to-portrait-3x4", action="store_true",
                        help="Pad internally; never pad the output MP4")
    parser.add_argument("--pad-value", type=int, default=127)
    parser.add_argument("--strict-eval-manifest")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.pad_value < 0 or args.pad_value > 255:
        parser.error("--pad-value must be within [0, 255]")
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("the output MP4 must not overwrite the input video")
    if args.max_detections is not None and args.max_detections <= 0:
        parser.error("--max-detections must be positive")
    return args


def frozen_value(args, frozen, name, default, frozen_key=None):
    requested = getattr(args, name)
    key = frozen_key or name
    if requested is not None:
        if key in frozen and requested != frozen[key]:
            raise ValueError(
                "%s=%s conflicts with frozen validation value %s" %
                (name, requested, frozen[key]))
        return requested
    return frozen.get(key, default)


def resolved_settings(args):
    frozen = {}
    if args.frozen_config:
        with open(args.frozen_config) as handle:
            frozen = json.load(handle)
    return {
        "stage1_score_threshold": float(frozen_value(
            args, frozen, "stage1_score_threshold", 0.50)),
        "stage2_score_threshold": float(frozen_value(
            args, frozen, "stage2_score_threshold", 0.70)),
        "nms_threshold": float(frozen_value(
            args, frozen, "nms_threshold", 0.30)),
        "max_detections": int(frozen_value(
            args, frozen, "max_detections", 1, "max_detections_final")),
        "crop_margin": float(frozen_value(
            args, frozen, "crop_margin", 0.20)),
        "stage2_min_geometry_iou": float(frozen_value(
            args, frozen, "stage2_min_geometry_iou", 0.20)),
    }


def draw_original_frame(frame, detections):
    rendered = frame.copy()
    height, width = frame.shape[:2]
    for detection_index, detection in enumerate(detections):
        corners = detection["ordered_corners"]
        integer_points = [(int(round(x)), int(round(y))) for x, y in corners]
        for index, point in enumerate(integer_points):
            following = integer_points[(index + 1) % len(integer_points)]
            cv2.line(rendered, point, following, (0, 255, 0), 2, cv2.LINE_AA)
        for index, point in enumerate(integer_points):
            color = CORNER_COLORS[index]
            cv2.circle(rendered, point, 4, color, -1, cv2.LINE_AA)
            label = "P%d (%d,%d)" % (index, point[0], point[1])
            size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            text_x = max(0, min(point[0] + 5, width - size[0] - 1))
            preferred_y = point[1] - 7 if index in (0, 1) else point[1] + 16
            text_y = max(size[1] + baseline, min(preferred_y, height - 2))
            cv2.putText(rendered, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(rendered, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,
                        1, cv2.LINE_AA)
        bbox = detection["derived_bbox_xyxy"]
        title = "QR%d S1=%.3f S2=%.3f" % (
            detection_index, detection["stage1_score"],
            detection["stage2_score"])
        label_y = max(16, int(round(bbox[1])) - 20)
        cv2.putText(rendered, title, (max(0, int(round(bbox[0]))), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0),
                    1, cv2.LINE_AA)
    return rendered


def write_csv_rows(writer, frame_index, fps, detections):
    for detection_index, detection in enumerate(detections):
        row = {
            "frame_index": frame_index,
            "time_seconds": frame_index / fps,
            "detection_index": detection_index,
            "score": detection["score"],
            "stage1_score": detection["stage1_score"],
            "stage2_score": detection["stage2_score"],
            "stage2_geometry_iou": detection["stage2_geometry_iou"],
        }
        for corner_index, point in enumerate(detection["ordered_corners"]):
            row["p%d_x" % corner_index] = point[0]
            row["p%d_y" % corner_index] = point[1]
        writer.writerow(row)


def verify_output_video(path, expected_width, expected_height, expected_frames):
    probe = cv2.VideoCapture(path)
    try:
        if not probe.isOpened():
            raise IOError("cannot reopen output video %s" % path)
        actual_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        probe.release()
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise RuntimeError(
            "output resolution %dx%d differs from original %dx%d" %
            (actual_width, actual_height, expected_width, expected_height))
    if actual_frames > 0 and actual_frames != expected_frames:
        raise RuntimeError("output has %d frames; expected %d" %
                           (actual_frames, expected_frames))
    return actual_width, actual_height, actual_frames


def main():
    args = parse_args()
    settings = resolved_settings(args)
    strict_item = None
    if args.strict_eval_manifest:
        strict_item = verify_final_input(args.strict_eval_manifest, args.input)

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise IOError("cannot open input video %s" % args.input)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 25.0
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("input video has invalid dimensions")

    initial_padding = (compute_center_padding(width, height, 3, 4, True)
                       if args.pad_to_portrait_3x4 else (0, 0, 0, 0))
    detector = TwoStageQRDetector(
        args.fsd_repo, args.stage1_checkpoint, args.stage2_checkpoint,
        args.device, settings["stage1_score_threshold"],
        settings["stage2_score_threshold"], settings["nms_threshold"],
        settings["max_detections"], settings["crop_margin"],
        settings["stage2_min_geometry_iou"], args.stage1_opencv_refine)

    output_stem = os.path.splitext(args.output)[0]
    jsonl_path = args.jsonl_output or output_stem + "_original_coords.jsonl"
    csv_path = args.csv_output or output_stem + "_original_coords.csv"
    summary_path = args.summary_output or output_stem + "_summary.json"
    for artifact in (jsonl_path, csv_path, summary_path):
        parent = os.path.dirname(os.path.abspath(artifact))
        if not os.path.isdir(parent):
            os.makedirs(parent)
    writer = create_writer(args.output, args.codec, fps, (width, height))
    records = open(jsonl_path, "w")
    table = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(table, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    frame_index = total_stage1 = total_ordered = rejected_count = 0
    rejected_padding_only = frames_with_ordered = 0
    started = time.time()
    try:
        while True:
            ok, original_frame = capture.read()
            if not ok:
                break
            if original_frame.shape[:2] != (height, width):
                raise RuntimeError("frame %d changed source resolution" % frame_index)
            inference_frame = original_frame
            padding = (0, 0, 0, 0)
            if args.pad_to_portrait_3x4:
                inference_frame, padding = pad_frame_to_portrait_3x4(
                    original_frame, args.pad_value)
            detections, rejected, stage1 = detector.predict(inference_frame, True)
            original_detections, discarded = original_frame_detections(
                detections, padding, width, height)
            total_stage1 += len(stage1)
            total_ordered += len(original_detections)
            rejected_count += len(rejected)
            rejected_padding_only += discarded
            frames_with_ordered += int(bool(original_detections))
            writer.write(draw_original_frame(original_frame, original_detections))
            record = {
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "coordinate_space": "original_video_pixels",
                "source_size": [width, height],
                "output_size": [width, height],
                "stage1_model_input_size": [240, 320],
                "stage2_model_input_size": [112, 112],
                "inference_frame_size": [
                    inference_frame.shape[1], inference_frame.shape[0]],
                "internal_padding": {
                    "left": padding[0], "top": padding[1],
                    "right": padding[2], "bottom": padding[3]},
                "stage1_count": len(stage1),
                "detections": original_detections,
                "stage2_rejected": rejected,
                "rejected_padding_only": discarded,
            }
            records.write(json.dumps(record, sort_keys=True) + "\n")
            write_csv_rows(csv_writer, frame_index, fps, original_detections)
            frame_index += 1
            if frame_index % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print("frames=%d fps=%.2f stage1=%d ordered=%d output=%dx%d" %
                      (frame_index, frame_index / elapsed, total_stage1,
                       total_ordered, width, height))
    finally:
        capture.release()
        writer.release()
        records.close()
        table.close()

    if frame_index == 0:
        raise RuntimeError("input video contained no decodable frames")
    actual_width, actual_height, actual_frames = verify_output_video(
        args.output, width, height, frame_index)
    elapsed = max(time.time() - started, 1e-6)
    summary = {
        "input": args.input,
        "output": args.output,
        "coordinates_jsonl": jsonl_path,
        "coordinates_csv": csv_path,
        "coordinate_space": "original_video_pixels",
        "source_size": [width, height],
        "output_size": [actual_width, actual_height],
        "source_fps": fps,
        "frames": frame_index,
        "verified_output_frames": actual_frames,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_index / elapsed,
        "stage1_model_input_size": [240, 320],
        "stage2_model_input_size": [112, 112],
        "stage1_detections": total_stage1,
        "ordered_detections": total_ordered,
        "stage2_rejections": rejected_count,
        "rejected_padding_only": rejected_padding_only,
        "frames_with_ordered_detection": frames_with_ordered,
        "internal_padding": {
            "enabled": bool(args.pad_to_portrait_3x4),
            "left": initial_padding[0], "top": initial_padding[1],
            "right": initial_padding[2], "bottom": initial_padding[3]},
        "frozen_config": args.frozen_config,
        "inference_settings": settings,
        "stage1_opencv_refine": bool(args.stage1_opencv_refine),
        "strict_final_evaluation": strict_item is not None,
        "strict_eval_video_sha256": strict_item["sha256"] if strict_item else None,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ORIGINAL_RESOLUTION_TWO_STAGE_INFERENCE_PASS")


if __name__ == "__main__":
    main()
