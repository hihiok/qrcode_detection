#!/usr/bin/env python3
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np

from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, SEMANTIC_CORNER_ORDER,
                       validate_semantic_corners)
from qr_dataset import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--visualize", type=int, default=32)
    args = parser.parse_args()
    output = os.path.join(args.data_root, "validation_preview")
    if args.visualize and not os.path.isdir(output):
        os.makedirs(output)
    total = 0
    for split in ("train", "val", "test"):
        split_root = os.path.join(args.data_root, split)
        rows = read_jsonl(os.path.join(split_root, "annotations.jsonl"))
        seen = set()
        for index, row in enumerate(rows):
            required = ("image", "width", "height", "corners", "corner_order")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError("%s[%d] missing %s" % (split, index, missing))
            if row["image"] in seen:
                raise ValueError("duplicate image in %s: %s" % (split, row["image"]))
            seen.add(row["image"])
            if row["width"] != INPUT_WIDTH or row["height"] != INPUT_HEIGHT:
                raise ValueError("%s must be W,H=240,320" % row["image"])
            if tuple(row["corner_order"]) != SEMANTIC_CORNER_ORDER:
                raise ValueError("%s must use QR-native semantic corner order" % row["image"])
            corners = validate_semantic_corners(row["corners"], row["image"])
            if ((corners[:, 0] < 0).any() or (corners[:, 0] >= INPUT_WIDTH).any() or
                    (corners[:, 1] < 0).any() or (corners[:, 1] >= INPUT_HEIGHT).any()):
                raise ValueError("%s corners are outside image" % row["image"])
            path = os.path.join(split_root, row["image"])
            image = cv2.imread(path)
            if image is None or image.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                raise ValueError("bad image/shape: %s" % path)
            if index < args.visualize:
                pts = np.rint(corners).astype(np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                colors = [(0, 0, 255), (0, 255, 255),
                          (255, 0, 0), (255, 0, 255)]
                for point_index, (x, y) in enumerate(pts):
                    cv2.circle(image, (int(x), int(y)), 4, colors[point_index], -1)
                    cv2.putText(image, "P%d" % point_index,
                                (int(x) + 3, int(y) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                colors[point_index], 1)
                cv2.imwrite(os.path.join(output, "%s_%04d.jpg" % (split, index)), image)
        print("%s: %d valid images" % (split, len(rows)))
        total += len(rows)
    print("PASS: %d single-QR images; semantic P0->P3 order preserved." % total)
    print("NOTE: geometry can validate clockwise order, but humans must verify P0 is QR-native TL.")


if __name__ == "__main__":
    main()
