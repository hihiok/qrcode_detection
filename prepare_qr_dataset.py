#!/usr/bin/env python3
"""Create a fixed 240x320 portrait, zero-or-more-QR dataset.

Subcommands:
  synthetic  - generate 0..N rendered QR instances over varied backgrounds
  negatives  - build explicit zero-QR splits from background images
  labelme    - convert every four-point QR polygon in each LabelMe image
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import string

import cv2
import numpy as np
import qrcode

from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, SEMANTIC_CORNER_ORDER,
                       corners_to_bbox, letterbox_image_points,
                       rotate_image_points_cw, validate_semantic_corners)
from qr_schema import SCHEMA_VERSION, canonical_json, validate_canonical_row


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def list_images(root):
    if not root:
        return []
    result = []
    for parent, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                result.append(os.path.join(parent, name))
    return sorted(result)


def write_jsonl(path, rows):
    with open(path, "w") as handle:
        for index, row in enumerate(rows, 1):
            validate_canonical_row(row, "%s:%d" % (path, index))
            handle.write(canonical_json(row) + "\n")


def random_payload(rng, index):
    alphabet = string.ascii_letters + string.digits + "-_:/?=&"
    length = int(rng.randint(8, 150))
    token = "".join(rng.choice(list(alphabet), size=length))
    return "fsd-qr-%08d-%s" % (index, token)


def render_qr(payload, rng):
    corrections = [qrcode.constants.ERROR_CORRECT_L,
                   qrcode.constants.ERROR_CORRECT_M,
                   qrcode.constants.ERROR_CORRECT_Q,
                   qrcode.constants.ERROR_CORRECT_H]
    qr = qrcode.QRCode(version=None,
                       error_correction=corrections[int(rng.randint(0, 4))],
                       box_size=int(rng.randint(3, 9)), border=4)
    qr.add_data(payload)
    qr.make(fit=True)
    dark = tuple(int(x) for x in rng.randint(0, 40, size=3))
    light_value = int(rng.randint(218, 256))
    light = (light_value, light_value, light_value)
    pil = qr.make_image(fill_color=dark, back_color=light).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def procedural_background(rng):
    h, w = INPUT_HEIGHT, INPUT_WIDTH
    base = rng.randint(20, 235, size=3).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    gx = rng.uniform(-80, 80, size=3)
    gy = rng.uniform(-80, 80, size=3)
    image = base + (xx[:, :, None] / float(w)) * gx + (yy[:, :, None] / float(h)) * gy
    image = np.clip(image, 0, 255).astype(np.uint8)
    for _ in range(int(rng.randint(5, 30))):
        color = tuple(int(x) for x in rng.randint(0, 256, size=3))
        if rng.rand() < 0.5:
            p1 = (int(rng.randint(0, w)), int(rng.randint(0, h)))
            p2 = (int(rng.randint(0, w)), int(rng.randint(0, h)))
            cv2.rectangle(image, p1, p2, color, int(rng.randint(1, 8)))
        else:
            center = (int(rng.randint(0, w)), int(rng.randint(0, h)))
            cv2.circle(image, center, int(rng.randint(3, 50)), color,
                       int(rng.choice([-1, 1, 2, 4])))
    return cv2.GaussianBlur(image, (3, 3), 0.4)


def cover_resize(image):
    h, w = image.shape[:2]
    scale = max(float(INPUT_WIDTH) / w, float(INPUT_HEIGHT) / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    left = max(0, (rw - INPUT_WIDTH) // 2)
    top = max(0, (rh - INPUT_HEIGHT) // 2)
    return resized[top:top + INPUT_HEIGHT, left:left + INPUT_WIDTH].copy()


def load_background(background_paths, rng, rotate_landscape_cw):
    if not background_paths or rng.rand() < 0.25:
        return procedural_background(rng)
    path = background_paths[int(rng.randint(0, len(background_paths)))]
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return procedural_background(rng)
    if rotate_landscape_cw and image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return cover_resize(image)


def sample_quad(rng, min_side, max_side):
    h, w = INPUT_HEIGHT, INPUT_WIDTH
    for _ in range(100):
        side = float(rng.uniform(min_side, max_side))
        half = side / 2.0
        angle = rng.uniform(-np.pi, np.pi)
        local = np.float32([[-half, -half], [half, -half],
                            [half, half], [-half, half]])
        rotation = np.float32([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        quad = np.dot(local, rotation.T)
        quad[:, 0] *= rng.uniform(0.78, 1.18)
        quad[:, 1] *= rng.uniform(0.78, 1.18)
        quad += rng.uniform(-0.12 * side, 0.12 * side, size=(4, 2))
        cx = rng.uniform(0.18 * w, 0.82 * w)
        cy = rng.uniform(0.14 * h, 0.86 * h)
        quad += np.float32([cx, cy])
        if ((quad[:, 0] >= 3).all() and (quad[:, 0] < w - 3).all() and
                (quad[:, 1] >= 3).all() and (quad[:, 1] < h - 3).all()):
            try:
                # local[] already defines QR-native TL,TR,BR,BL. Rotation changes
                # where P0 appears in the image but must never change its index.
                return validate_semantic_corners(quad, "synthetic quad")
            except ValueError:
                continue
    raise RuntimeError("Cannot sample in-frame QR quadrilateral")


def paste_qr(background, qr_image, quad):
    qh, qw = qr_image.shape[:2]
    source = np.float32([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]])
    matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped = cv2.warpPerspective(qr_image, matrix, (INPUT_WIDTH, INPUT_HEIGHT),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    mask_source = np.full((qh, qw), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_source, matrix, (INPUT_WIDTH, INPUT_HEIGHT),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(background * (1.0 - alpha) + warped * alpha, 0, 255).astype(np.uint8)


def add_decoy_patterns(image, rng):
    """Add non-QR square/grid/text structures used as hard negatives."""
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(int(rng.randint(1, 5))):
        mode = int(rng.randint(0, 4))
        color = tuple(int(v) for v in rng.randint(0, 256, size=3))
        if mode == 0:
            side = int(rng.randint(10, 55))
            x = int(rng.randint(0, max(1, w - side)))
            y = int(rng.randint(0, max(1, h - side)))
            cells = int(rng.randint(2, 7))
            cell = max(2, side // cells)
            for row in range(cells):
                for column in range(cells):
                    if (row + column + int(rng.randint(0, 2))) % 2 == 0:
                        cv2.rectangle(out, (x + column * cell, y + row * cell),
                                      (min(w - 1, x + (column + 1) * cell),
                                       min(h - 1, y + (row + 1) * cell)), color, -1)
        elif mode == 1:
            side = int(rng.randint(12, 65))
            x = int(rng.randint(0, max(1, w - side)))
            y = int(rng.randint(0, max(1, h - side)))
            for inset in range(0, max(2, side // 2), max(2, side // 7)):
                cv2.rectangle(out, (x + inset, y + inset),
                              (x + side - inset, y + side - inset), color, 1)
        elif mode == 2:
            step = int(rng.randint(6, 25))
            for x in range(int(rng.randint(0, step)), w, step):
                cv2.line(out, (x, 0), (x, h - 1), color, 1)
            for y in range(int(rng.randint(0, step)), h, step):
                cv2.line(out, (0, y), (w - 1, y), color, 1)
        else:
            text = "".join(str(int(v)) for v in rng.randint(0, 10, size=12))
            cv2.putText(out, text, (int(rng.randint(0, max(1, w // 3))),
                                    int(rng.randint(20, h))),
                        cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.35, 0.8),
                        color, int(rng.choice([1, 2])))
    return out


def motion_blur(image, rng):
    length = int(rng.choice([3, 5, 7, 9]))
    kernel = np.zeros((length, length), np.float32)
    cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1.0, 1)
    angle = float(rng.uniform(0.0, 180.0))
    matrix = cv2.getRotationMatrix2D((length / 2.0 - 0.5,
                                      length / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def simulate_yuv420(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    h, w = yuv.shape[:2]
    for channel in (1, 2):
        small = cv2.resize(yuv[:, :, channel],
                           (max(1, w // 2), max(1, h // 2)),
                           interpolation=cv2.INTER_AREA)
        yuv[:, :, channel] = cv2.resize(small, (w, h),
                                        interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def degrade(image, quads, rng, allow_decoys=False):
    out = image.copy()
    if allow_decoys and rng.rand() < 0.65:
        out = add_decoy_patterns(out, rng)
    # Partial occlusion is kept small so all four geometric corners remain valid.
    if quads and rng.rand() < 0.18:
        quad = quads[int(rng.randint(0, len(quads)))]
        box = corners_to_bbox(quad)
        bw, bh = box[2] - box[0], box[3] - box[1]
        ow = max(2, int(bw * rng.uniform(0.04, 0.16)))
        oh = max(2, int(bh * rng.uniform(0.04, 0.16)))
        x = int(rng.uniform(box[0] + 0.2 * bw, max(box[0] + 0.2 * bw + 1,
                                                   box[2] - ow)))
        y = int(rng.uniform(box[1] + 0.2 * bh, max(box[1] + 0.2 * bh + 1,
                                                   box[3] - oh)))
        color = tuple(int(v) for v in rng.randint(0, 256, size=3))
        cv2.rectangle(out, (x, y), (x + ow, y + oh), color, -1)
    if rng.rand() < 0.35:
        kernel = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (kernel, kernel), rng.uniform(0.2, 1.6))
    if rng.rand() < 0.20:
        out = motion_blur(out, rng)
    if rng.rand() < 0.25:
        factor = float(rng.uniform(0.35, 0.8))
        small = cv2.resize(out, None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (image.shape[1], image.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    if rng.rand() < 0.20:
        gamma = float(rng.uniform(0.55, 1.65))
        lookup = np.asarray([min(255, round((value / 255.0) ** gamma * 255.0))
                             for value in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lookup)
    if rng.rand() < 0.18:
        overlay = out.copy()
        center = (int(rng.randint(0, INPUT_WIDTH)),
                  int(rng.randint(0, INPUT_HEIGHT)))
        axes = (int(rng.randint(20, 100)), int(rng.randint(8, 45)))
        cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360,
                    (255, 255, 255), -1)
        out = cv2.addWeighted(out, 1.0, overlay, rng.uniform(0.08, 0.32), 0)
    if rng.rand() < 0.15:
        yy, xx = np.mgrid[0:INPUT_HEIGHT, 0:INPUT_WIDTH]
        pattern = np.sin(xx * rng.uniform(0.25, 0.8) +
                         yy * rng.uniform(0.10, 0.45))[:, :, None]
        out = np.clip(out.astype(np.float32) +
                      pattern * rng.uniform(2.0, 12.0), 0, 255).astype(np.uint8)
    if rng.rand() < 0.35:
        out = simulate_yuv420(out)
    alpha = rng.uniform(0.55, 1.35)
    beta = rng.uniform(-35, 28)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255)
    out += rng.normal(0.0, rng.uniform(0, 7), size=out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.rand() < 0.55:
        quality = int(rng.randint(30, 96))
        ok, encoded = cv2.imencode(".jpg", out,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out


def bbox_iou_numpy(a, b):
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-9)


def sample_nonoverlapping_quad(rng, existing, min_side, max_side, max_iou=0.10):
    for _ in range(200):
        candidate = sample_quad(rng, min_side, max_side)
        box = corners_to_bbox(candidate)
        if all(bbox_iou_numpy(box, corners_to_bbox(other)) <= max_iou
               for other in existing):
            return candidate
    raise RuntimeError("Cannot place another non-overlapping QR")


def generate_split(root, split, count, seed, background_paths,
                   rotate_landscape_cw, min_side, max_side,
                   max_qrs, negative_ratio, single_qr_probability,
                   decoy_probability):
    split_root = os.path.join(root, split)
    image_root = os.path.join(split_root, "images")
    mkdir(image_root)
    rng = np.random.RandomState(seed)
    rows = []
    for index in range(count):
        background = load_background(background_paths, rng, rotate_landscape_cw)
        if rng.rand() < negative_ratio:
            num_qrs = 0
        elif rng.rand() < single_qr_probability:
            num_qrs = 1
        else:
            num_qrs = int(rng.randint(1, max_qrs + 1))
        quads = []
        image = background
        for qr_index in range(num_qrs):
            try:
                quad = sample_nonoverlapping_quad(
                    rng, quads, min_side, max_side)
            except RuntimeError:
                break
            qr_image = render_qr(
                random_payload(rng, index * max_qrs + qr_index), rng)
            image = paste_qr(image, qr_image, quad)
            quads.append(quad)
        image = degrade(image, quads, rng,
                        allow_decoys=(num_qrs == 0 and
                                      rng.rand() < decoy_probability))
        name = "%s_%07d.jpg" % (split, index)
        relative = os.path.join("images", name)
        if not cv2.imwrite(os.path.join(split_root, relative), image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise IOError("Failed to write %s" % name)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "num_qrcodes": len(quads),
            "instances": [
                {"class_id": 0, "label": "qrcode",
                 "corners": [[float(x), float(y)] for x, y in quad],
                 "corner_order": list(SEMANTIC_CORNER_ORDER)}
                for quad in quads
            ],
            "metadata": {"synthetic": True,
                         "source_kind": ("synthetic_negative" if not quads
                                         else "synthetic_positive")},
        })
        if (index + 1) % 1000 == 0:
            print("%s: %d/%d" % (split, index + 1, count))
    write_jsonl(os.path.join(split_root, "annotations.jsonl"), rows)
    print("Wrote %s: %d images" % (split_root, len(rows)))


def command_synthetic(args):
    root = os.path.abspath(args.output)
    mkdir(root)
    backgrounds = list_images(args.background_dir)
    background_splits = {"train": [], "val": [], "test": []}
    for path in backgrounds:
        name = os.path.relpath(path, args.background_dir)
        split = split_from_name(name, args.background_train_ratio,
                                args.background_val_ratio)
        background_splits[split].append(path)
    print("Background images: total=%d train=%d val=%d test=%d" %
          (len(backgrounds), len(background_splits["train"]),
           len(background_splits["val"]), len(background_splits["test"])))
    generate_split(root, "train", args.train_count, args.seed,
                   background_splits["train"], args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio,
                   args.single_qr_probability, args.decoy_probability)
    generate_split(root, "val", args.val_count, args.seed + 1000003,
                   background_splits["val"], args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio,
                   args.single_qr_probability, args.decoy_probability)
    generate_split(root, "test", args.test_count, args.seed + 2000003,
                   background_splits["test"], args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio,
                   args.single_qr_probability, args.decoy_probability)


def command_negatives(args):
    paths = list_images(args.input)
    if args.max_count > 0:
        paths = paths[:args.max_count]
    if not paths:
        raise RuntimeError("No negative candidate images found in %s" % args.input)
    rows = {"train": [], "val": [], "test": []}
    for split in rows:
        mkdir(os.path.join(args.output, split, "images"))
    rng = np.random.RandomState(args.seed)
    for index, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cover_resize(image)
        image = degrade(image, [], rng,
                        allow_decoys=(rng.rand() < args.decoy_probability))
        relative_source = os.path.relpath(path, args.input)
        split = split_from_name(relative_source, args.train_ratio, args.val_ratio)
        name = "negative_%07d.jpg" % index
        relative = os.path.join("images", name)
        output_path = os.path.join(args.output, split, relative)
        if not cv2.imwrite(output_path, image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise IOError("Failed to write %s" % output_path)
        rows[split].append({
            "schema_version": SCHEMA_VERSION,
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "num_qrcodes": 0,
            "instances": [],
            "metadata": {"synthetic": False, "source_kind": "negative",
                         "source": relative_source},
        })
    for split, split_rows in rows.items():
        write_jsonl(os.path.join(args.output, split, "annotations.jsonl"), split_rows)
        print("%s: %d negatives" % (split, len(split_rows)))


def find_labelme_image(json_path, data):
    candidate = data.get("imagePath")
    if candidate:
        path = os.path.join(os.path.dirname(json_path), candidate)
        if os.path.isfile(path):
            return path
    stem = os.path.splitext(json_path)[0]
    for extension in IMAGE_EXTENSIONS:
        if os.path.isfile(stem + extension):
            return stem + extension
    raise IOError("Cannot resolve image for %s" % json_path)


def split_from_name(name, train_ratio, val_ratio):
    value = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) / float(0xffffffff)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def command_labelme(args):
    json_paths = []
    for parent, _, names in os.walk(args.input):
        for name in names:
            if name.lower().endswith(".json"):
                json_paths.append(os.path.join(parent, name))
    json_paths.sort()
    rows = {"train": [], "val": [], "test": []}
    for split in rows:
        mkdir(os.path.join(args.output, split, "images"))
    for index, json_path in enumerate(json_paths):
        with open(json_path, "r") as handle:
            data = json.load(handle)
        shapes = [s for s in data.get("shapes", [])
                  if s.get("label", "").lower() in ("qr", "qrcode", "qr_code")]
        points = np.empty((0, 4, 2), dtype=np.float32)
        if shapes:
            points = np.stack([
                validate_semantic_corners(
                    shape.get("points", []), "%s QR shape %d" % (json_path, i))
                for i, shape in enumerate(shapes)]).astype(np.float32)
        image = cv2.imread(find_labelme_image(json_path, data), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Cannot read image for %s" % json_path)
        rotated = False
        if args.rotate_cw == "always" or (args.rotate_cw == "landscape" and
                                           image.shape[1] > image.shape[0]):
            original_shape = points.shape
            image, points = rotate_image_points_cw(image, points.reshape(-1, 2))
            points = points.reshape(original_shape)
            rotated = True
        original_shape = points.shape
        image, points, letterbox = letterbox_image_points(
            image, points.reshape(-1, 2))
        points = points.reshape(original_shape)
        if len(points):
            points = np.stack([
                validate_semantic_corners(
                    quad, "%s transformed instance %d" % (json_path, i))
                for i, quad in enumerate(points)])
        if (len(points) and
                ((points[:, :, 0] < 0).any() or
                 (points[:, :, 0] >= INPUT_WIDTH).any() or
                 (points[:, :, 1] < 0).any() or
                 (points[:, :, 1] >= INPUT_HEIGHT).any())):
            raise ValueError("%s corners leave final 240x320 image" % json_path)
        split = split_from_name(os.path.relpath(json_path, args.input),
                                args.train_ratio, args.val_ratio)
        name = "real_%07d.jpg" % index
        relative = os.path.join("images", name)
        cv2.imwrite(os.path.join(args.output, split, relative), image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rows[split].append({
            "schema_version": SCHEMA_VERSION,
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "num_qrcodes": len(points),
            "instances": [
                {"class_id": 0, "label": "qrcode",
                 "corners": [[float(x), float(y)] for x, y in quad],
                 "corner_order": list(SEMANTIC_CORNER_ORDER)}
                for quad in points
            ],
            "metadata": {
                "synthetic": False,
                "source": os.path.relpath(json_path, args.input),
                "rotated_clockwise": rotated,
                "letterbox": letterbox,
            },
        })
    for split, split_rows in rows.items():
        write_jsonl(os.path.join(args.output, split, "annotations.jsonl"), split_rows)
        print("%s: %d" % (split, len(split_rows)))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--background-dir", default=None)
    synthetic.add_argument("--train-count", type=int, default=40000)
    synthetic.add_argument("--val-count", type=int, default=4000)
    synthetic.add_argument("--test-count", type=int, default=4000)
    synthetic.add_argument("--seed", type=int, default=20260805)
    synthetic.add_argument("--min-qr-side", type=float, default=28.0)
    synthetic.add_argument("--max-qr-side", type=float, default=190.0)
    synthetic.add_argument("--max-qrs-per-image", type=int, default=5)
    synthetic.add_argument("--negative-ratio", type=float, default=0.15)
    synthetic.add_argument("--single-qr-probability", type=float, default=0.90)
    synthetic.add_argument("--decoy-probability", type=float, default=0.65)
    synthetic.add_argument("--background-train-ratio", type=float, default=0.80)
    synthetic.add_argument("--background-val-ratio", type=float, default=0.10)
    synthetic.add_argument("--rotate-landscape-cw", action="store_true")
    synthetic.set_defaults(function=command_synthetic)

    negatives = sub.add_parser("negatives")
    negatives.add_argument("--input", required=True)
    negatives.add_argument("--output", required=True)
    negatives.add_argument("--max-count", type=int, default=0)
    negatives.add_argument("--seed", type=int, default=20260818)
    negatives.add_argument("--train-ratio", type=float, default=0.80)
    negatives.add_argument("--val-ratio", type=float, default=0.10)
    negatives.add_argument("--decoy-probability", type=float, default=0.35)
    negatives.set_defaults(function=command_negatives)

    labelme = sub.add_parser("labelme")
    labelme.add_argument("--input", required=True)
    labelme.add_argument("--output", required=True)
    labelme.add_argument("--rotate-cw", choices=("never", "landscape", "always"),
                         default="landscape")
    labelme.add_argument("--train-ratio", type=float, default=0.85)
    labelme.add_argument("--val-ratio", type=float, default=0.10)
    labelme.set_defaults(function=command_labelme)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed = parser.parse_args()
    if not hasattr(parsed, "function"):
        parser.print_help()
        raise SystemExit(2)
    parsed.function(parsed)
