#!/usr/bin/env python3
"""Run YUV multi-QR detection on MP4/video and preserve all NMS detections."""
from __future__ import print_function

import argparse
import json
import os
import time

import cv2

from infer_fsd_qr import QRDetector, draw_results


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
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()

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
    else:
        output_size = (width, height)
    parent = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*args.codec), fps, output_size)
    if not writer.isOpened():
        raise IOError("Cannot create video: %s" % args.output)

    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, 240,
        args.score_threshold, args.nms_threshold, 400, args.max_detections)
    jsonl_path = os.path.splitext(args.output)[0] + ".jsonl"
    jsonl_handle = open(jsonl_path, "w")
    frame_index = 0
    started = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rotated = False
        if args.rotate_landscape_cw and frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rotated = True
        detections = detector.predict(frame)
        writer.write(draw_results(frame, detections))
        record = {
            "frame_index": frame_index,
            "time_seconds": frame_index / fps,
            "rotated_clockwise": rotated,
            "detections": detections}
        jsonl_handle.write(json.dumps(record) + "\n")
        frame_index += 1
        if frame_index % 100 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print("%d frames, %.2f FPS, current detections=%d" %
                  (frame_index, frame_index / elapsed, len(detections)))
    capture.release()
    writer.release()
    jsonl_handle.close()
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump({
            "input": args.input, "output": args.output, "fps": fps,
            "num_frames": frame_index, "detections_jsonl": jsonl_path},
            handle, indent=2)
    print("Wrote %s, %s and %s (%d frames)" %
          (args.output, jsonl_path, summary_path, frame_index))


if __name__ == "__main__":
    main()
