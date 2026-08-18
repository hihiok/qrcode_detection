#!/usr/bin/env python3
"""Mine false-positive candidates without touching frozen business videos."""
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np

from infer_fsd_qr import QRDetector, draw_results, list_images
from prepare_qr_dataset import mkdir, split_from_name, write_jsonl
from qr_common import INPUT_HEIGHT, INPUT_WIDTH, letterbox_image_points
from qr_schema import SCHEMA_VERSION
from strict_eval_guard import (contains_forbidden_token, load_manifest,
                               normalized_stem)


def ensure_not_strict_eval_candidate(path, manifest):
    lowered = os.path.realpath(path).lower()
    for item in manifest["videos"]:
        for token in (item["basename"].lower(), normalized_stem(item["path"])):
            if len(token) >= 4 and contains_forbidden_token(lowered, token):
                raise ValueError("Strict-eval content offered to miner: %s" % path)


def opencv_finds_qr(image):
    detector = cv2.QRCodeDetector()
    try:
        found, points = detector.detect(image)
        return bool(found and points is not None)
    except cv2.error:
        return False


def make_previews(items, output_dir, page_size=100, columns=4,
                  tile_width=360, tile_height=240):
    outputs = []
    for page_start in range(0, len(items), page_size):
        page = items[page_start:page_start + page_size]
        rows = int(np.ceil(len(page) / float(columns)))
        canvas = np.full((max(rows, 1) * tile_height,
                          columns * tile_width, 3), 127, dtype=np.uint8)
        for local_index, (image, detections, label) in enumerate(page):
            drawn = draw_results(image, detections)
            scale = min(tile_width / float(drawn.shape[1]),
                        tile_height / float(drawn.shape[0]))
            resized = cv2.resize(
                drawn, (max(1, int(drawn.shape[1] * scale)),
                        max(1, int(drawn.shape[0] * scale))))
            tile = np.full((tile_height, tile_width, 3), 127, np.uint8)
            tile[:resized.shape[0], :resized.shape[1]] = resized
            cv2.putText(tile, label[:48], (3, tile_height - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            row, column = divmod(local_index, columns)
            canvas[row * tile_height:(row + 1) * tile_height,
                   column * tile_width:(column + 1) * tile_width] = tile
        output = os.path.join(
            output_dir, "hard_negative_preview_%03d.jpg" %
            (page_start // page_size))
        cv2.imwrite(output, canvas)
        outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-eval-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--exclude-source-file",
                        help="Optional text file: one relative source path to exclude per line")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()
    strict_manifest = load_manifest(args.strict_eval_manifest,
                                    verify_video_hashes=True)
    if os.path.exists(args.output):
        raise ValueError("Output already exists: %s" % args.output)
    paths = list_images(args.input)
    excluded = set()
    if args.exclude_source_file:
        with open(args.exclude_source_file, "r") as handle:
            excluded = set(line.strip().replace("\\", "/") for line in handle
                           if line.strip() and not line.lstrip().startswith("#"))
    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, 240,
        args.score_threshold, 0.30, 400, 20)
    candidates = []
    for index, path in enumerate(paths):
        ensure_not_strict_eval_candidate(path, strict_manifest)
        relative_candidate = os.path.relpath(path, args.input).replace("\\", "/")
        if relative_candidate in excluded:
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        detections = detector.predict(image)
        if detections and not opencv_finds_qr(image):
            candidates.append((max(item["score"] for item in detections),
                               path, image, detections))
        if (index + 1) % 500 == 0:
            print("scanned %d/%d candidates=%d" %
                  (index + 1, len(paths), len(candidates)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:args.max_candidates]
    if not candidates:
        raise RuntimeError("No hard-negative candidates found")
    rows = {"train": [], "val": [], "test": []}
    for split in rows:
        mkdir(os.path.join(args.output, split, "images"))
    preview_items = []
    review_rows = []
    for index, (score, path, image, detections) in enumerate(candidates):
        relative_source = os.path.relpath(path, args.input)
        split = split_from_name(relative_source, args.train_ratio, args.val_ratio)
        transformed, _, transform = letterbox_image_points(
            image, np.empty((0, 2), np.float32), INPUT_WIDTH, INPUT_HEIGHT)
        name = "hard_negative_%07d.jpg" % index
        relative = os.path.join("images", name)
        cv2.imwrite(os.path.join(args.output, split, relative), transformed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        row = {
            "schema_version": SCHEMA_VERSION,
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "num_qrcodes": 0,
            "instances": [],
            "metadata": {"source_kind": "hard_negative_candidate",
                         "source": relative_source,
                         "mining_score": float(score),
                         "letterbox": transform},
        }
        rows[split].append(row)
        review_rows.append({"rank": index, "score": float(score),
                            "source": relative_source, "split": split,
                            "output_image": os.path.join(split, relative)})
        if len(preview_items) < args.preview_count:
            preview_items.append((image, detections,
                                  "%04d %.3f %s" %
                                  (index, score, os.path.basename(path))))
    for split, values in rows.items():
        write_jsonl(os.path.join(args.output, split, "annotations.jsonl"), values)
    preview_paths = make_previews(preview_items, args.output)
    with open(os.path.join(args.output, "HARD_NEGATIVE_REVIEW.json"), "w") as handle:
        json.dump({"review_required": True,
                   "instruction": "Remove any real QR image, then create a non-empty APPROVED_BY_HUMAN.txt",
                   "preview_pages": preview_paths,
                   "candidates": review_rows}, handle, indent=2)
    print("Wrote %d candidates. HUMAN REVIEW IS REQUIRED before manifest build." %
          len(candidates))


if __name__ == "__main__":
    main()
