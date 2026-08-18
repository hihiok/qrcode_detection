#!/usr/bin/env python3
"""Run YUV multi-QR detection on video with optional exact 3:4 padding."""
from __future__ import print_function

import argparse
import json
import os
import time

import cv2

from infer_fsd_qr import QRDetector, draw_results
from video_padding import compute_center_padding, padded_size


def pad_frame_to_portrait_3x4(frame, pad_value=127):
    height, width = frame.shape[:2]
    padding = compute_center_padding(width, height, 3, 4, True)
    left, top, right, bottom = padding
    padded = cv2.copyMakeBorder(
        frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value))
    return padded, padding


def create_writer(path, codec, fps, output_size):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*codec), fps, output_size)
    if not writer.isOpened():
        raise IOError("Cannot create video: %s" % path)
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--rotate-landscape-cw", action="store_true")
    parser.add_argument(
        "--pad-to-portrait-3x4", action="store_true",
        help="Center-pad without stretching before the model resizes to 240x320")
    parser.add_argument(
        "--pad-value", type=int, default=127,
        help="Constant BGR padding value in [0,255]; default is neutral gray")
    parser.add_argument(
        "--padded-input-output",
        help="Optional unannotated 3:4 padded video written from model input frames")
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()

    if args.rotate_landscape_cw and args.pad_to_portrait_3x4:
        raise ValueError("rotation and 3:4 padding are mutually exclusive")
    if args.padded_input_output and not args.pad_to_portrait_3x4:
        raise ValueError("--padded-input-output requires --pad-to-portrait-3x4")
    if args.pad_value < 0 or args.pad_value > 255:
        raise ValueError("--pad-value must be in [0,255]")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("input and output video paths must differ")
    if (args.padded_input_output and
            os.path.abspath(args.input) == os.path.abspath(args.padded_input_output)):
        raise ValueError("padded input output must not overwrite the source video")
    if (args.padded_input_output and
            os.path.abspath(args.output) == os.path.abspath(args.padded_input_output)):
        raise ValueError("annotated and unannotated output paths must differ")

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise IOError("Cannot open video: %s" % args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.rotate_landscape_cw and width > height:
        output_size = (height, width)
        initial_padding = (0, 0, 0, 0)
    elif args.pad_to_portrait_3x4:
        initial_padding = compute_center_padding(width, height, 3, 4, True)
        output_size = padded_size(width, height, initial_padding)
    else:
        output_size = (width, height)
        initial_padding = (0, 0, 0, 0)

    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, 240,
        args.score_threshold, args.nms_threshold, 400, args.max_detections)
    writer = create_writer(args.output, args.codec, fps, output_size)
    padded_writer = None
    if args.padded_input_output:
        padded_writer = create_writer(
            args.padded_input_output, args.codec, fps, output_size)
    jsonl_path = os.path.splitext(args.output)[0] + ".jsonl"
    jsonl_handle = open(jsonl_path, "w")
    frame_index = 0
    total_detections = 0
    frames_with_detections = 0
    max_detections_in_frame = 0
    started = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_height, source_width = frame.shape[:2]
            rotated = False
            padding = (0, 0, 0, 0)
            if args.rotate_landscape_cw and source_width > source_height:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                rotated = True
            elif args.pad_to_portrait_3x4:
                frame, padding = pad_frame_to_portrait_3x4(frame, args.pad_value)
            if (frame.shape[1], frame.shape[0]) != output_size:
                raise ValueError(
                    "frame %d produced %dx%d but writer expects %dx%d" %
                    (frame_index, frame.shape[1], frame.shape[0],
                     output_size[0], output_size[1]))
            if padded_writer is not None:
                padded_writer.write(frame)
            detections = detector.predict(frame)
            detection_count = len(detections)
            total_detections += detection_count
            frames_with_detections += int(detection_count > 0)
            max_detections_in_frame = max(
                max_detections_in_frame, detection_count)
            writer.write(draw_results(frame, detections))
            left, top, right, bottom = padding
            record = {
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "source_size": [source_width, source_height],
                "inference_frame_size": [frame.shape[1], frame.shape[0]],
                "coordinate_space": (
                    "padded_portrait_3x4" if args.pad_to_portrait_3x4
                    else "output_frame"),
                "padding": {"left": left, "top": top,
                            "right": right, "bottom": bottom},
                "rotated_clockwise": rotated,
                "detections": detections}
            jsonl_handle.write(json.dumps(record) + "\n")
            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print("%d frames, %.2f FPS, current detections=%d" %
                      (frame_index, frame_index / elapsed, len(detections)))
    finally:
        capture.release()
        writer.release()
        if padded_writer is not None:
            padded_writer.release()
        jsonl_handle.close()

    elapsed_seconds = max(time.time() - started, 1e-6)
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    left, top, right, bottom = initial_padding
    with open(summary_path, "w") as handle:
        json.dump({
            "input": args.input,
            "output": args.output,
            "padded_input_output": args.padded_input_output,
            "fps": fps,
            "num_frames": frame_index,
            "elapsed_seconds": elapsed_seconds,
            "average_processing_fps": frame_index / elapsed_seconds,
            "total_detections": total_detections,
            "frames_with_detections": frames_with_detections,
            "max_detections_in_frame": max_detections_in_frame,
            "source_size": [width, height],
            "output_size": list(output_size),
            "pad_to_portrait_3x4": args.pad_to_portrait_3x4,
            "padding": {"left": left, "top": top,
                        "right": right, "bottom": bottom},
            "detections_jsonl": jsonl_path}, handle, indent=2)
    print("Wrote %s, %s and %s (%d frames)" %
          (args.output, jsonl_path, summary_path, frame_index))


if __name__ == "__main__":
    main()
