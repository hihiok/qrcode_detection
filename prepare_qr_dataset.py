#!/usr/bin/env python3
"""Create a fixed 240x320 portrait, zero-or-more-QR dataset.

Subcommands:
  synthetic  - generate 0..N rendered QR instances over varied backgrounds
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
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def degrade(image, quads, rng):
    out = image.copy()
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
        length = int(rng.choice([3, 5, 7]))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        out = cv2.filter2D(out, -1, kernel)
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
                   max_qrs, negative_ratio):
    split_root = os.path.join(root, split)
    image_root = os.path.join(split_root, "images")
    mkdir(image_root)
    rng = np.random.RandomState(seed)
    rows = []
    for index in range(count):
        background = load_background(background_paths, rng, rotate_landscape_cw)
        num_qrs = (0 if rng.rand() < negative_ratio else
                   int(rng.randint(1, max_qrs + 1)))
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
        image = degrade(image, quads, rng)
        name = "%s_%07d.jpg" % (split, index)
        relative = os.path.join("images", name)
        if not cv2.imwrite(os.path.join(split_root, relative), image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise IOError("Failed to write %s" % name)
        rows.append({
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "instances": [
                {"label": "qrcode",
                 "corners": [[float(x), float(y)] for x, y in quad],
                 "corner_order": list(SEMANTIC_CORNER_ORDER)}
                for quad in quads
            ],
            "synthetic": True,
        })
        if (index + 1) % 1000 == 0:
            print("%s: %d/%d" % (split, index + 1, count))
    write_jsonl(os.path.join(split_root, "annotations.jsonl"), rows)
    print("Wrote %s: %d images" % (split_root, len(rows)))


def command_synthetic(args):
    root = os.path.abspath(args.output)
    mkdir(root)
    backgrounds = list_images(args.background_dir)
    print("Background images: %d" % len(backgrounds))
    generate_split(root, "train", args.train_count, args.seed,
                   backgrounds, args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio)
    generate_split(root, "val", args.val_count, args.seed + 1000003,
                   backgrounds, args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio)
    generate_split(root, "test", args.test_count, args.seed + 2000003,
                   backgrounds, args.rotate_landscape_cw,
                   args.min_qr_side, args.max_qr_side,
                   args.max_qrs_per_image, args.negative_ratio)


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
            "image": relative.replace(os.sep, "/"),
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "instances": [
                {"label": "qrcode",
                 "corners": [[float(x), float(y)] for x, y in quad],
                 "corner_order": list(SEMANTIC_CORNER_ORDER)}
                for quad in points
            ],
            "synthetic": False,
            "source": os.path.relpath(json_path, args.input),
            "rotated_clockwise": rotated,
            "letterbox": letterbox,
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
    synthetic.add_argument("--rotate-landscape-cw", action="store_true")
    synthetic.set_defaults(function=command_synthetic)

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
