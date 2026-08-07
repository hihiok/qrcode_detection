#!/usr/bin/env python3
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np

from qr_common import INPUT_HEIGHT, INPUT_WIDTH
from qr_dataset import instances_from_row, read_jsonl


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
    for split in ("train", "val", "test"):
        split_root = os.path.join(args.data_root, split)
        rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))
        seen = set()
        split_instances = split_negatives = 0
        for index, row in enumerate(rows):
            required = ("image", "width", "height")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError("%s[%d] missing %s" % (split, index, missing))
            if row["image"] in seen:
                raise ValueError("duplicate image in %s: %s" % (split, row["image"]))
            seen.add(row["image"])
            if row["width"] != INPUT_WIDTH or row["height"] != INPUT_HEIGHT:
                raise ValueError("%s must be W,H=240,320" % row["image"])
            corners = instances_from_row(row, row["image"])
            if len(corners) > args.max_instances:
                raise ValueError("%s has %d instances; max=%d" %
                                 (row["image"], len(corners), args.max_instances))
            if (len(corners) and
                    ((corners[:, :, 0] < 0).any() or
                     (corners[:, :, 0] >= INPUT_WIDTH).any() or
                     (corners[:, :, 1] < 0).any() or
                     (corners[:, :, 1] >= INPUT_HEIGHT).any())):
                raise ValueError("%s corners are outside image" % row["image"])
            path = os.path.join(split_root, row["image"])
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
        print("%s: %d images, %d QR instances, %d negative images" %
              (split, len(rows), split_instances, split_negatives))
        total_images += len(rows)
        total_instances += split_instances
        negative_images += split_negatives
    print("PASS: %d images, %d QR instances, %d negatives." %
          (total_images, total_instances, negative_images))
    print("NOTE: humans must verify every instance keeps QR-native P0->P3 identity.")


if __name__ == "__main__":
    main()
