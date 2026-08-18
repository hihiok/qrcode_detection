#!/usr/bin/env python3
"""Canonical ordered-corner QR annotation schema and legacy conversion.

This module is deliberately dependency-free so dataset conversion and schema
checks can run before importing OpenCV, NumPy, PyTorch, or the model code.
"""
from __future__ import print_function

import json
import math
import os


SCHEMA_VERSION = "qr_ordered_corners_v1"
CORNER_ORDER = [
    "qr_top_left",
    "qr_top_right",
    "qr_bottom_right",
    "qr_bottom_left",
]
SUPPORTED_SOURCE_KEYS = ("instances", "objects", "corners")


def read_jsonl(path):
    rows = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc))
    return rows


def _name(name, suffix):
    return "%s %s" % (name, suffix) if suffix else name


def _safe_relative_path(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty relative path" % name)
    normalized = os.path.normpath(value)
    if os.path.isabs(value) or normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("%s must stay inside its split directory: %s" % (name, value))
    return value


def _validate_corner_order(value, name, allow_missing):
    if value is None and allow_missing:
        return list(CORNER_ORDER)
    if value != CORNER_ORDER:
        raise ValueError("%s corner_order must be exactly %s; got %s" %
                         (name, CORNER_ORDER, value))
    return list(CORNER_ORDER)


def _validate_corners(value, width, height, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("%s corners must contain exactly four points" % name)
    result = []
    for point_index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("%s P%d must be [x,y]" % (name, point_index))
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            raise ValueError("%s P%d coordinates must be numeric" % (name, point_index))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("%s P%d coordinates must be finite" % (name, point_index))
        if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
            raise ValueError("%s P%d is outside %dx%d: [%s,%s]" %
                             (name, point_index, width, height, x, y))
        result.append([x, y])
    if len(set((point[0], point[1]) for point in result)) != 4:
        raise ValueError("%s corners must be four distinct points" % name)
    twice_area = 0.0
    for index in range(4):
        x1, y1 = result[index]
        x2, y2 = result[(index + 1) % 4]
        twice_area += x1 * y2 - x2 * y1
    if abs(twice_area) < 1e-3:
        raise ValueError("%s corners form a degenerate polygon" % name)
    return result


def _canonical_instance(value, width, height, name, allow_missing_order):
    if not isinstance(value, dict):
        value = {"corners": value}
    if "corners" not in value:
        raise ValueError("%s is missing corners" % name)
    label = value.get("label", "qrcode")
    class_id = int(value.get("class_id", 0))
    if label != "qrcode" or class_id != 0:
        raise ValueError("%s must be label=qrcode and class_id=0" % name)
    instance = {
        "class_id": 0,
        "label": "qrcode",
        "corners": _validate_corners(value["corners"], width, height, name),
        "corner_order": _validate_corner_order(
            value.get("corner_order"), name, allow_missing_order),
    }
    extras = dict((key, item) for key, item in value.items()
                  if key not in ("class_id", "label", "corners", "corner_order"))
    if extras:
        instance["metadata"] = extras
    return instance


def _source_values(row, source_schema, name):
    if source_schema == "instances":
        values = row["instances"]
        if not isinstance(values, list):
            raise ValueError("%s instances must be a list" % name)
        return values
    if source_schema == "objects":
        values = row["objects"]
        if not isinstance(values, list):
            raise ValueError("%s objects must be a list" % name)
        return values
    corners = row["corners"]
    if (isinstance(corners, (list, tuple)) and len(corners) == 4 and
            all(isinstance(point, (list, tuple)) and len(point) == 2
                for point in corners)):
        return [{"label": row.get("label", "qrcode"),
                 "class_id": row.get("class_id", 0),
                 "corners": corners,
                 "corner_order": row.get("corner_order")}]
    if isinstance(corners, list):
        return [{"label": "qrcode", "class_id": 0, "corners": value,
                 "corner_order": row.get("corner_order")} for value in corners]
    raise ValueError("%s top-level corners has an unsupported shape" % name)


def canonicalize_row(row, dataset_name, split, line_number):
    """Convert one known legacy row without changing corner coordinates/order."""
    name = "%s/%s annotations.jsonl:%d" % (dataset_name, split, line_number)
    if not isinstance(row, dict):
        raise ValueError("%s must be a JSON object" % name)
    width = int(row.get("width", -1))
    height = int(row.get("height", -1))
    if (width, height) != (240, 320):
        raise ValueError("%s must declare width=240 height=320" % name)
    image = _safe_relative_path(row.get("image"), _name(name, "image"))
    present = [key for key in SUPPORTED_SOURCE_KEYS if key in row]
    if len(present) != 1:
        raise ValueError("%s must contain exactly one of %s; got %s" %
                         (name, SUPPORTED_SOURCE_KEYS, present))
    source_schema = present[0]
    values = _source_values(row, source_schema, name)
    # BarBeR instances predate explicit corner_order but were already manually
    # validated as QR-native P0/P1/P2/P3. Other source formats must declare it.
    allow_missing_order = source_schema == "instances"
    instances = [
        _canonical_instance(value, width, height,
                            "%s instance %d" % (name, index), allow_missing_order)
        for index, value in enumerate(values)
    ]
    if "num_qrcodes" in row and int(row["num_qrcodes"]) != len(instances):
        raise ValueError("%s num_qrcodes=%s but parsed %d instances" %
                         (name, row["num_qrcodes"], len(instances)))
    top_level_excluded = set(("schema_version", "image", "width", "height",
                              "num_qrcodes", "instances", "objects", "corners",
                              "label", "class_id", "corner_order"))
    metadata = dict((key, value) for key, value in row.items()
                    if key not in top_level_excluded)
    metadata.update({
        "dataset": dataset_name,
        "split": split,
        "source_schema": source_schema,
    })
    result = {
        "schema_version": SCHEMA_VERSION,
        "image": image,
        "width": width,
        "height": height,
        "num_qrcodes": len(instances),
        "instances": instances,
        "metadata": metadata,
    }
    validate_canonical_row(result, name)
    return result


def validate_canonical_row(row, name="annotation"):
    if not isinstance(row, dict):
        raise ValueError("%s must be a JSON object" % name)
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("%s schema_version must be %s" % (name, SCHEMA_VERSION))
    image = _safe_relative_path(row.get("image"), _name(name, "image"))
    width = int(row.get("width", -1))
    height = int(row.get("height", -1))
    if (width, height) != (240, 320):
        raise ValueError("%s must declare width=240 height=320" % name)
    if "instances" not in row or not isinstance(row["instances"], list):
        raise ValueError("%s instances must be an explicit list" % name)
    instances = [
        _canonical_instance(value, width, height,
                            "%s instance %d" % (name, index), False)
        for index, value in enumerate(row["instances"])
    ]
    if "num_qrcodes" not in row:
        raise ValueError("%s is missing num_qrcodes" % name)
    if int(row["num_qrcodes"]) != len(instances):
        raise ValueError("%s num_qrcodes=%s but instances=%d" %
                         (name, row["num_qrcodes"], len(instances)))
    return image, width, height, instances


def canonical_json(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
