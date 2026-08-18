#!/usr/bin/env python3
"""Audit QR V2 labels, priors, YUV conversion and coordinate round trips."""
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np

from dataset_v2_manifest import load_and_verify_manifest
from infer_fsd_qr import letterbox, restore_points
from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, corners_to_bbox,
                       generate_portrait_priors, match_qr_instances)
from qr_dataset import bgr_to_yuv_tensor, instances_from_row
from qr_schema import read_jsonl


def size_bucket(side):
    if side < 32:
        return "lt32"
    if side < 64:
        return "32_63"
    if side < 128:
        return "64_127"
    return "ge128"


def coordinate_roundtrip_audit():
    cases = [(480, 408), (1280, 720), (240, 320), (320, 240), (641, 479)]
    result = []
    for width, height in cases:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        _, meta = letterbox(image)
        points = np.asarray([[0.10 * width, 0.15 * height],
                             [0.80 * width, 0.12 * height],
                             [0.78 * width, 0.82 * height],
                             [0.13 * width, 0.85 * height]], np.float32)
        normalized = points.copy()
        normalized[:, 0] = (normalized[:, 0] * meta["scale"] + meta["left"]) / INPUT_WIDTH
        normalized[:, 1] = (normalized[:, 1] * meta["scale"] + meta["top"]) / INPUT_HEIGHT
        restored = restore_points(normalized, meta)
        error = float(np.abs(restored - points).max())
        result.append({"source_size": [width, height], "max_abs_error_px": error,
                       "letterbox": meta})
        if error > 1e-3:
            raise ValueError("Coordinate round trip failed for %dx%d: %.6f" %
                             (width, height, error))
    return result


def yuv_audit():
    image = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = 17
    image[:, :, 1] = 91
    image[:, :, 2] = 203
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).transpose(2, 0, 1) / 255.0
    actual = bgr_to_yuv_tensor(image).numpy()
    error = float(np.abs(actual - expected).max())
    if actual.shape != (3, INPUT_HEIGHT, INPUT_WIDTH) or error > 1e-7:
        raise ValueError("BGR->YUV audit failed shape=%s error=%g" %
                         (actual.shape, error))
    return {"shape": list(actual.shape), "max_abs_error": error,
            "conversion": "cv2.COLOR_BGR2YUV, YUV444, float32/255"}


def draw_preview(items, output, columns=5, cell_width=240, cell_height=320):
    if not items:
        return None
    rows = int(np.ceil(len(items) / float(columns)))
    canvas = np.full((rows * cell_height, columns * cell_width, 3),
                     127, dtype=np.uint8)
    for index, (image, corners, label) in enumerate(items):
        tile = image.copy()
        for instance_index, quad in enumerate(corners):
            pts = np.rint(quad).astype(np.int32)
            cv2.polylines(tile, [pts], True, (0, 255, 0), 2)
            for point_index, point in enumerate(pts):
                cv2.circle(tile, tuple(point), 3, (0, 0, 255), -1)
                cv2.putText(tile, "P%d" % point_index,
                            (int(point[0]) + 2, int(point[1]) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1)
        cv2.putText(tile, label[:32], (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 255), 1)
        row, column = divmod(index, columns)
        canvas[row * cell_height:(row + 1) * cell_height,
               column * cell_width:(column + 1) * cell_width] = tile
    cv2.imwrite(output, canvas)
    return output


def inspect_source(source, split, priors, max_images, preview_limit):
    split_root = os.path.join(source["root"], split)
    rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))
    if max_images > 0:
        rows = rows[:max_images]
    stats = {"images": len(rows), "instances": 0, "negative_images": 0,
             "positive_anchors": [], "size_buckets": {}, "corner_bounds_failures": 0}
    preview = []
    for row in rows:
        path = os.path.join(split_root, row["image"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Cannot read %s" % path)
        corners = instances_from_row(row, path)
        stats["instances"] += len(corners)
        stats["negative_images"] += int(len(corners) == 0)
        if ((corners[:, :, 0] < 0).any() or (corners[:, :, 0] >= INPUT_WIDTH).any() or
                (corners[:, :, 1] < 0).any() or
                (corners[:, :, 1] >= INPUT_HEIGHT).any()):
            stats["corner_bounds_failures"] += 1
        normalized = corners / np.asarray([INPUT_WIDTH, INPUT_HEIGHT], np.float32)
        labels, _, _ = match_qr_instances(normalized, priors, 0.35)
        stats["positive_anchors"].append(int((labels > 0).sum().item()))
        for quad in corners:
            box = corners_to_bbox(quad)
            side = float(np.sqrt(max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))))
            bucket = size_bucket(side)
            stats["size_buckets"][bucket] = stats["size_buckets"].get(bucket, 0) + 1
        if len(preview) < preview_limit:
            preview.append((image, corners, "%s/%s" % (source["name"], row["image"])))
    anchors = stats.pop("positive_anchors")
    stats["positive_anchors_min"] = min(anchors) if anchors else 0
    stats["positive_anchors_mean"] = float(np.mean(anchors)) if anchors else 0.0
    stats["positive_anchors_max"] = max(anchors) if anchors else 0
    if stats["corner_bounds_failures"]:
        raise ValueError("Out-of-bounds corners in source %s" % source["name"])
    return stats, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-images-per-source", type=int, default=0)
    parser.add_argument("--preview-per-source", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = load_and_verify_manifest(args.manifest)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    priors, shapes = generate_portrait_priors()
    report = {
        "schema_version": "qr_training_pipeline_audit_v1",
        "manifest": os.path.realpath(args.manifest),
        "split": args.split,
        "num_priors": int(priors.size(0)),
        "feature_shapes": shapes,
        "coordinate_roundtrip": coordinate_roundtrip_audit(),
        "yuv": yuv_audit(),
        "sources": {},
    }
    previews = []
    for source in manifest["sources"]:
        stats, items = inspect_source(source, args.split, priors,
                                      args.max_images_per_source,
                                      args.preview_per_source)
        report["sources"][source["name"]] = stats
        previews.extend(items)
    preview_path = os.path.join(args.output_dir, "ordered_corner_preview.jpg")
    report["preview"] = draw_preview(previews, preview_path)
    report_path = os.path.join(args.output_dir, "pipeline_audit.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
