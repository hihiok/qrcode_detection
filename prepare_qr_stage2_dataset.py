#!/usr/bin/env python3
"""Build frozen per-instance ROI data for the second QR orientation stage."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import cv2
import numpy as np

from dataset_v2_manifest import load_and_verify_manifest
from qr_common import SEMANTIC_CORNER_ORDER, validate_semantic_corners
from qr_schema import canonical_json, read_jsonl, validate_canonical_row
from qr_stage2_dataset import (STAGE2_SCHEMA_VERSION, STAGE2_SIZE,
                               read_rows as read_stage2_rows)
from qr_two_stage_geometry import order_quad_for_crop, transform_points, warp_qr_roi
from strict_eval_guard import check_dataset_roots


MANIFEST_NAME = "QR_STAGE2_DATASET_MANIFEST.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_sources(values):
    result = []
    seen = set()
    for value in values or []:
        if "=" not in value:
            raise ValueError("--source must be NAME=/absolute/root: %s" % value)
        name, root = value.split("=", 1)
        name = name.strip()
        root = os.path.realpath(os.path.abspath(root.strip()))
        if not name or name in seen:
            raise ValueError("invalid or duplicate source name: %s" % name)
        if not os.path.isdir(root):
            raise IOError("source does not exist: %s" % root)
        seen.add(name)
        result.append((name, root))
    return result


def sources_from_args(args):
    direct = parse_named_sources(args.source)
    if args.dataset_manifest and direct:
        raise ValueError("use --dataset-manifest or --source, not both")
    if args.dataset_manifest:
        manifest = load_and_verify_manifest(args.dataset_manifest)
        return [(item["name"], item["root"]) for item in manifest["sources"]], \
            os.path.realpath(args.dataset_manifest)
    if not direct:
        raise ValueError("one --dataset-manifest or at least one --source is required")
    return direct, None


def safe_token(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_"
                   for ch in str(value))


def make_row(relative_image, corners, metadata):
    instances = []
    if corners is not None:
        validated = validate_semantic_corners(corners, relative_image)
        instances.append({
            "class_id": 0,
            "label": "qrcode",
            "corners": [[float(x), float(y)] for x, y in validated],
            "corner_order": list(SEMANTIC_CORNER_ORDER),
        })
    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "image": relative_image,
        "width": STAGE2_SIZE,
        "height": STAGE2_SIZE,
        "num_qrcodes": len(instances),
        "instances": instances,
        "metadata": metadata,
    }


def positive_crop(image, semantic_corners, rng, margin_min, margin_max, jitter,
                  training):
    geometry = order_quad_for_crop(semantic_corners)
    for _ in range(30):
        margin = (float(rng.uniform(margin_min, margin_max)) if training
                  else 0.5 * (float(margin_min) + float(margin_max)))
        use_jitter = float(jitter) if training else 0.0
        roi, crop_quad, matrix, _ = warp_qr_roi(
            image, geometry, margin, rng if training else None, use_jitter,
            output_width=STAGE2_SIZE, output_height=STAGE2_SIZE)
        transformed = transform_points(semantic_corners, matrix).reshape(4, 2)
        in_frame = bool(
            (transformed[:, 0] >= 0.0).all() and
            (transformed[:, 0] < STAGE2_SIZE).all() and
            (transformed[:, 1] >= 0.0).all() and
            (transformed[:, 1] < STAGE2_SIZE).all())
        if in_frame:
            try:
                validate_semantic_corners(transformed, "stage2 transformed")
                return roi, transformed, crop_quad, matrix, margin
            except ValueError:
                pass
    roi, crop_quad, matrix, _ = warp_qr_roi(
        image, geometry, max(float(margin_max), 0.25), None, 0.0,
        output_width=STAGE2_SIZE, output_height=STAGE2_SIZE)
    transformed = transform_points(semantic_corners, matrix).reshape(4, 2)
    validate_semantic_corners(transformed, "stage2 fallback transformed")
    return roi, transformed, crop_quad, matrix, max(float(margin_max), 0.25)


def write_image(path, image):
    ok = cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise IOError("failed to write %s" % path)


def negative_square_crop(image, rng, training, size=STAGE2_SIZE):
    height, width = image.shape[:2]
    maximum = min(height, width)
    side = (int(round(rng.uniform(0.35, 0.95) * maximum))
            if training else int(round(0.75 * maximum)))
    side = max(8, min(side, maximum))
    if training:
        left = int(rng.randint(0, width - side + 1))
        top = int(rng.randint(0, height - side + 1))
    else:
        left = (width - side) // 2
        top = (height - side) // 2
    crop = image[top:top + side, left:left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def inspect_stage2_split(root, split):
    split_root = os.path.join(root, split)
    annotation_path = os.path.join(split_root, "annotations.jsonl")
    rows = read_stage2_rows(annotation_path)
    instances = negatives = 0
    image_digest = hashlib.sha256()
    for row in rows:
        path = os.path.join(split_root, row["image"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (STAGE2_SIZE, STAGE2_SIZE):
            raise ValueError("bad stage2 image: %s" % path)
        image_digest.update(row["image"].encode("utf-8"))
        image_digest.update(sha256_file(path).encode("ascii"))
        count = len(row["instances"])
        instances += count
        negatives += int(count == 0)
    return {"images": len(rows), "instances": instances,
            "negative_images": negatives,
            "annotations_sha256": sha256_file(annotation_path),
            "images_sha256": image_digest.hexdigest(),
            "annotation_path": os.path.realpath(annotation_path)}


def write_previews(root, max_items=100, page_size=20):
    split_root = os.path.join(root, "train")
    rows = [row for row in read_jsonl(os.path.join(split_root, "annotations.jsonl"))
            if row.get("instances")]
    if not rows:
        raise RuntimeError("cannot preview an empty positive stage-2 dataset")
    count = min(int(max_items), len(rows))
    indices = np.linspace(0, len(rows) - 1, count).astype(np.int64)
    preview_paths = []
    preview_index = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for page_start in range(0, count, int(page_size)):
        tiles = []
        for tile_index, index in enumerate(
                indices[page_start:page_start + int(page_size)]):
            row = rows[int(index)]
            image = cv2.imread(os.path.join(split_root, row["image"]),
                               cv2.IMREAD_COLOR)
            corners = np.asarray(row["instances"][0]["corners"], np.int32)
            cv2.polylines(image, [corners], True, (255, 255, 255), 2)
            for point_index, (x, y) in enumerate(corners):
                cv2.circle(image, (int(x), int(y)), 5,
                           colors[point_index], -1)
                cv2.putText(image, "P%d" % point_index,
                            (int(x) + 4, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            colors[point_index], 2)
            tiles.append(image)
            preview_index.append({
                "page": page_start // int(page_size) + 1,
                "tile": tile_index,
                "dataset_image": row["image"],
                "metadata": row.get("metadata", {}),
            })
        while len(tiles) < int(page_size):
            tiles.append(np.full_like(tiles[0], 127))
        columns = 4
        rows_per_page = int(page_size) // columns
        page = np.vstack([
            np.hstack(tiles[row_index * columns:(row_index + 1) * columns])
            for row_index in range(rows_per_page)])
        path = os.path.join(
            root, "stage2_preview_%02d.jpg" % (page_start // int(page_size) + 1))
        write_image(path, page)
        preview_paths.append(path)
    index_path = os.path.join(root, "stage2_preview_index.json")
    with open(index_path, "w") as handle:
        json.dump(preview_index, handle, indent=2, sort_keys=True)
    return preview_paths, index_path


def build_split(work_root, split, sources, variants, args, rng):
    split_root = os.path.join(work_root, split)
    image_root = os.path.join(split_root, "images")
    os.makedirs(image_root)
    annotation_path = os.path.join(split_root, "annotations.jsonl")
    stats = {"positive_images": 0, "negative_images": 0,
             "source_instances": 0, "by_source": {}}
    with open(annotation_path, "w") as output:
        for source_name, source_root in sources:
            source_annotation = os.path.join(source_root, split, "annotations.jsonl")
            rows = read_jsonl(source_annotation)
            source_stats = {"rows": len(rows), "instances": 0,
                            "positive_crops": 0, "negative_crops": 0}
            for row_index, row in enumerate(rows):
                relative, _, _, instances = validate_canonical_row(
                    row, "%s:%d" % (source_annotation, row_index + 1))
                source_image = os.path.join(source_root, split, relative)
                image = cv2.imread(source_image, cv2.IMREAD_COLOR)
                if image is None:
                    raise IOError("cannot read %s" % source_image)
                if image.shape[:2] != (320, 240):
                    raise ValueError("canonical source must be 240x320: %s" %
                                     source_image)
                source_stats["instances"] += len(instances)
                stats["source_instances"] += len(instances)
                for instance_index, instance in enumerate(instances):
                    semantic = np.asarray(instance["corners"], np.float32)
                    for variant in range(int(variants)):
                        roi, transformed, crop_quad, matrix, margin = positive_crop(
                            image, semantic, rng, args.margin_min, args.margin_max,
                            args.jitter, split == "train")
                        name = "%s_%s_r%07d_i%03d_v%02d.jpg" % (
                            safe_token(source_name), split, row_index,
                            instance_index, variant)
                        relative_out = os.path.join("images", name)
                        write_image(os.path.join(split_root, relative_out), roi)
                        metadata = {
                            "task": "stage2_semantic_ordered_corners",
                            "source_name": source_name,
                            "source_split": split,
                            "source_image": relative,
                            "source_instance_index": instance_index,
                            "variant": variant,
                            "crop_margin": float(margin),
                            "crop_quad_source_xy": crop_quad.tolist(),
                            "source_to_stage2_homography": matrix.tolist(),
                            "negative": False,
                        }
                        output.write(canonical_json(
                            make_row(relative_out, transformed, metadata)) + "\n")
                        stats["positive_images"] += 1
                        source_stats["positive_crops"] += 1
                if not instances and int(args.negative_per_image) > 0:
                    count = int(args.negative_per_image) if split == "train" else 1
                    for negative_index in range(count):
                        # Square-to-square background crop; no aspect distortion.
                        roi = negative_square_crop(
                            image, rng, split == "train")
                        name = "%s_%s_r%07d_n%02d.jpg" % (
                            safe_token(source_name), split, row_index, negative_index)
                        relative_out = os.path.join("images", name)
                        write_image(os.path.join(split_root, relative_out), roi)
                        metadata = {
                            "task": "stage2_semantic_ordered_corners",
                            "source_name": source_name,
                            "source_split": split,
                            "source_image": relative,
                            "negative": True,
                        }
                        output.write(canonical_json(
                            make_row(relative_out, None, metadata)) + "\n")
                        stats["negative_images"] += 1
                        source_stats["negative_crops"] += 1
            stats["by_source"][source_name] = source_stats
    if stats["positive_images"] == 0:
        raise RuntimeError("%s contains no stage-2 positive crops" % split)
    return stats


def build_dataset(args):
    sources, source_manifest = sources_from_args(args)
    output_root = os.path.realpath(os.path.abspath(args.output))
    if os.path.exists(output_root):
        raise ValueError("output already exists; refusing to overwrite: %s" % output_root)
    if args.margin_min < 0.05 or args.margin_max < args.margin_min:
        raise ValueError("require 0.05 <= margin-min <= margin-max")
    if args.jitter < 0.0 or args.jitter > 0.15:
        raise ValueError("--jitter must be in [0,0.15]")
    source_roots = [root for _, root in sources]
    check_dataset_roots(args.strict_eval_manifest, source_roots)
    parent = os.path.dirname(output_root)
    os.makedirs(parent, exist_ok=True)
    work_root = tempfile.mkdtemp(prefix=".qr_stage2_build_", dir=parent)
    rng = np.random.RandomState(args.seed)
    try:
        split_stats = {}
        for split, variants in (("train", args.train_variants),
                                ("val", args.val_variants),
                                ("test", args.test_variants)):
            split_stats[split] = build_split(
                work_root, split, sources, variants, args, rng)
        os.rename(work_root, output_root)
        work_root = None
    finally:
        if work_root and os.path.isdir(work_root):
            shutil.rmtree(work_root)
    check_dataset_roots(args.strict_eval_manifest, [output_root])
    preview_paths, preview_index = write_previews(output_root)
    frozen_stats = {}
    for split in ("train", "val", "test"):
        frozen_stats[split] = inspect_stage2_split(output_root, split)
    manifest = {
        "schema_version": "qr_stage2_dataset_manifest_v1",
        "purpose": "single-QR ROI semantic P0/P1/P2/P3 regression",
        "output_root": output_root,
        "strict_eval_manifest": os.path.realpath(args.strict_eval_manifest),
        "source_dataset_manifest": source_manifest,
        "sources": [{"name": name, "root": root} for name, root in sources],
        "config": {
            "train_variants": args.train_variants,
            "val_variants": args.val_variants,
            "test_variants": args.test_variants,
            "negative_per_image": args.negative_per_image,
            "margin_min": args.margin_min,
            "margin_max": args.margin_max,
            "jitter": args.jitter,
            "seed": args.seed,
            "output_size": [STAGE2_SIZE, STAGE2_SIZE],
            "corner_order": list(SEMANTIC_CORNER_ORDER),
        },
        "generation_stats": split_stats,
        "frozen_stats": frozen_stats,
        "human_review": {
            "status": "WAITING_FOR_HUMAN_STAGE2_PREVIEW_REVIEW",
            "preview_paths": preview_paths,
            "preview_index": preview_index,
            "required_approval_file": os.path.join(
                output_root, "APPROVED_STAGE2_DATASET.txt"),
        },
    }
    manifest_path = os.path.join(output_root, MANIFEST_NAME)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    manifest["manifest_path"] = manifest_path
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def verify_dataset(path):
    manifest_path = (path if os.path.isfile(path)
                     else os.path.join(path, MANIFEST_NAME))
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "qr_stage2_dataset_manifest_v1":
        raise ValueError("unsupported stage-2 manifest: %s" % manifest_path)
    root = manifest["output_root"]
    check_dataset_roots(manifest["strict_eval_manifest"],
                        [root] + [item["root"] for item in manifest["sources"]])
    for split in ("train", "val", "test"):
        actual = inspect_stage2_split(root, split)
        expected = manifest["frozen_stats"][split]
        for key in ("images", "instances", "negative_images",
                    "annotations_sha256", "images_sha256"):
            if actual[key] != expected[key]:
                raise ValueError("stage-2 dataset changed: %s/%s" % (split, key))
    return {"pass": True, "manifest": os.path.realpath(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "frozen_stats": manifest["frozen_stats"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build")
    build.add_argument("--dataset-manifest")
    build.add_argument("--source", action="append",
                       help="NAME=/absolute/canonical/dataset/root")
    build.add_argument("--strict-eval-manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--train-variants", type=int, default=4)
    build.add_argument("--val-variants", type=int, default=1)
    build.add_argument("--test-variants", type=int, default=1)
    build.add_argument("--negative-per-image", type=int, default=1)
    build.add_argument("--margin-min", type=float, default=0.12)
    build.add_argument("--margin-max", type=float, default=0.30)
    build.add_argument("--jitter", type=float, default=0.04)
    build.add_argument("--seed", type=int, default=20260821)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_dataset(args)
    elif args.command == "verify":
        result = verify_dataset(args.manifest)
    else:
        parser.print_help()
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
