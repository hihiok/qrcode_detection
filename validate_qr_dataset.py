#!/usr/bin/env python3
from __future__ import print_function

import argparse
import collections
import json
import os

import cv2
import numpy as np

from qr_common import INPUT_HEIGHT, INPUT_WIDTH
from qr_dataset import instances_from_row
from qr_schema import read_jsonl, validate_canonical_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--visualize", type=int, default=32)
    parser.add_argument("--max-instances", type=int, default=20)
    args = parser.parse_args()
    output = os.path.join(args.data_root, "validation_preview")
    if args.visualize and not os.path.isdir(output):
        os.makedirs(output)
    total_images = total_instances = negative_images = 0
    group_splits = {}
    report = {"data_root": os.path.abspath(args.data_root), "splits": {}}
    for split in ("train", "val", "test"):
        split_root = os.path.join(args.data_root, split)
        annotation_path = os.path.join(split_root, "annotations.jsonl")
        rows = read_jsonl(annotation_path)
        seen = set()
        split_instances = split_negatives = 0
        histogram = collections.Counter()
        for index, row in enumerate(rows):
            row_name = "%s:%d" % (annotation_path, index + 1)
            image_name, width, height, _ = validate_canonical_row(row, row_name)
            if image_name in seen:
                raise ValueError("duplicate image in %s: %s" % (split, image_name))
            seen.add(image_name)
            if width != INPUT_WIDTH or height != INPUT_HEIGHT:
                raise ValueError("%s must be W,H=240,320" % image_name)
            corners = instances_from_row(row, row_name)
            if len(corners) > args.max_instances:
                raise ValueError("%s has %d instances; max=%d" %
                                 (image_name, len(corners), args.max_instances))
            metadata = row.get("metadata", {})
            group_key = metadata.get("group_key") if isinstance(metadata, dict) else None
            if group_key is not None:
                previous = group_splits.get(str(group_key))
                if previous is not None and previous != split:
                    raise ValueError("group_key %s crosses %s and %s" %
                                     (group_key, previous, split))
                group_splits[str(group_key)] = split
            path = os.path.join(split_root, image_name)
            image = cv2.imread(path)
            if image is None or image.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                raise ValueError("bad image/shape: %s" % path)
            if index < args.visualize:
                for instance_index, quad in enumerate(corners):
                    pts = np.rint(quad).astype(np.int32)
                    color = tuple(int(v) for v in np.random.RandomState(
                        instance_index + 17).randint(64, 256, size=3))
                    cv2.polylines(image, [pts], True, color, 2)
                    for point_index, (x, y) in enumerate(pts):
                        cv2.circle(image, (int(x), int(y)), 3, color, -1)
                        cv2.putText(image, "%d:P%d" % (instance_index, point_index),
                                    (int(x) + 3, int(y) - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                cv2.imwrite(os.path.join(
                    output, "%s_%04d_%dqr.jpg" % (split, index, len(corners))), image)
            split_instances += len(corners)
            split_negatives += int(len(corners) == 0)
            histogram[str(len(corners))] += 1
        print("%s: %d images, %d QR instances, %d negative images" %
              (split, len(rows), split_instances, split_negatives))
        report["splits"][split] = {
            "images": len(rows), "instances": split_instances,
            "negative_images": split_negatives,
            "instances_per_image": dict(sorted(histogram.items()))}
        total_images += len(rows)
        total_instances += split_instances
        negative_images += split_negatives
    report.update({"images": total_images, "instances": total_instances,
                   "negative_images": negative_images,
                   "group_keys_checked": len(group_splits)})
    with open(os.path.join(args.data_root, "validation_report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("PASS: %d images, %d QR instances, %d negatives." %
          (total_images, total_instances, negative_images))
    print("NOTE: humans must verify every instance keeps QR-native P0->P3 identity.")


if __name__ == "__main__":
    main()
