#!/usr/bin/env python3
"""Convert the four QR datasets into one strict, non-destructive schema."""
from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os
import shutil

from qr_schema import (CORNER_ORDER, canonical_json, canonicalize_row,
                       read_jsonl, validate_canonical_row)


SPLITS = ("train", "val", "test")
SPLIT_PRIORITY = {"train": 1, "val": 2, "test": 3}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_dataset(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be NAME=/absolute/path")
    name, path = value.split("=", 1)
    name = name.strip().lower()
    path = os.path.abspath(path.strip())
    if not name or not path:
        raise argparse.ArgumentTypeError("--dataset must be NAME=/absolute/path")
    return name, path


def safe_join(root, relative, name):
    candidate = os.path.abspath(os.path.join(root, relative))
    root_prefix = os.path.abspath(root) + os.sep
    if not candidate.startswith(root_prefix):
        raise ValueError("%s escapes split root: %s" % (name, relative))
    return candidate


def verify_label_file(row, source_row, split_root, tolerance, name):
    label_file = source_row.get("label_file")
    if label_file is None:
        return False
    label_path = safe_join(split_root, label_file, "%s label_file" % name)
    if not os.path.isfile(label_path):
        raise ValueError("%s label_file does not exist: %s" % (name, label_path))
    parsed = []
    with open(label_path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 9 or int(fields[0]) != 0:
                raise ValueError("%s:%d must be: 0 + eight normalized coordinates" %
                                 (label_path, line_number))
            coords = [float(value) for value in fields[1:]]
            parsed.append([[coords[index] * row["width"],
                            coords[index + 1] * row["height"]]
                           for index in range(0, 8, 2)])
    if len(parsed) != len(row["instances"]):
        raise ValueError("%s label rows=%d but JSON instances=%d" %
                         (name, len(parsed), len(row["instances"])))
    for instance_index, (label_corners, instance) in enumerate(
            zip(parsed, row["instances"])):
        for point_index, (label_point, json_point) in enumerate(
                zip(label_corners, instance["corners"])):
            if (abs(label_point[0] - json_point[0]) > tolerance * row["width"] or
                    abs(label_point[1] - json_point[1]) > tolerance * row["height"]):
                raise ValueError(
                    "%s instance %d P%d disagrees with label_file beyond tolerance %g" %
                    (name, instance_index, point_index, tolerance))
    return True


def write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def label_signature(row):
    """Return the exact semantic label payload, excluding path/source metadata."""
    return canonical_json({
        "width": row["width"],
        "height": row["height"],
        "num_qrcodes": row["num_qrcodes"],
        "instances": row["instances"],
    })


def convert_dataset(name, source_root, output_root, tolerance, hash_images,
                    deduplicate_identical_images, global_image_hashes,
                    global_duplicate_warnings):
    if deduplicate_identical_images and not hash_images:
        raise ValueError("deduplicate-identical-images requires image hashing")
    dataset_output = os.path.join(output_root, name)
    os.makedirs(dataset_output)
    report = {"source_root": source_root, "splits": {},
              "corner_order": list(CORNER_ORDER), "group_leakage_checked": True,
              "duplicate_policy": (
                  "identical_sha256_and_labels_keep_test_then_val_then_train"
                  if deduplicate_identical_images else "reject_cross_split"),
              "duplicate_resolutions": []}
    group_splits = {}
    prepared_by_split = {}
    source_annotation_hashes = {}
    dataset_hash_groups = collections.defaultdict(list)

    # Read and validate every split before writing output. This is required to
    # choose a deterministic keeper when a later split outranks an earlier one.
    for split in SPLITS:
        source_split = os.path.join(source_root, split)
        annotation_path = os.path.join(source_split, "annotations.jsonl")
        image_root = os.path.join(source_split, "images")
        if not os.path.isfile(annotation_path) or not os.path.isdir(image_root):
            raise ValueError("%s/%s is missing annotations.jsonl or images/" %
                             (name, split))
        annotation_hash_before = sha256_file(annotation_path)
        source_annotation_hashes[split] = annotation_hash_before
        output_split = os.path.join(dataset_output, split)
        os.makedirs(output_split)
        os.symlink(os.path.abspath(image_root), os.path.join(output_split, "images"))
        labels_root = os.path.join(source_split, "labels")
        if os.path.isdir(labels_root):
            os.symlink(os.path.abspath(labels_root), os.path.join(output_split, "labels"))
        source_rows = read_jsonl(annotation_path)
        prepared_rows = []
        seen_images = set()
        for index, source_row in enumerate(source_rows, 1):
            row_name = "%s/%s:%d" % (name, split, index)
            row = canonicalize_row(source_row, name, split, index)
            image = row["image"]
            if image in seen_images:
                raise ValueError("duplicate image row in %s: %s" % (row_name, image))
            seen_images.add(image)
            image_path = safe_join(source_split, image, "%s image" % row_name)
            if not os.path.isfile(image_path):
                raise ValueError("%s image does not exist: %s" % (row_name, image_path))
            label_file_checked = verify_label_file(
                row, source_row, source_split, tolerance, row_name)
            group_key = source_row.get("group_key")
            if group_key is not None:
                previous = group_splits.get(str(group_key))
                if previous is not None and previous != split:
                    raise ValueError("%s group_key %s crosses %s and %s" %
                                     (name, group_key, previous, split))
                group_splits[str(group_key)] = split
            validate_canonical_row(row, row_name)
            image_hash = sha256_file(image_path) if hash_images else None
            prepared = {
                "split": split,
                "row_name": row_name,
                "row": row,
                "image_path": image_path,
                "image_hash": image_hash,
                "label_file_checked": label_file_checked,
                "dropped": False,
            }
            prepared_rows.append(prepared)
            if image_hash is not None:
                dataset_hash_groups[image_hash].append(prepared)
        prepared_by_split[split] = prepared_rows

    if hash_images:
        for image_hash, items in sorted(dataset_hash_groups.items()):
            if len(set(item["split"] for item in items)) <= 1:
                continue
            first = items[0]
            if len(set(label_signature(item["row"]) for item in items)) != 1:
                raise ValueError(
                    "%s exact image duplicate has conflicting canonical labels: %s" %
                    (name, ", ".join(item["image_path"] for item in items)))
            if not deduplicate_identical_images:
                raise ValueError("%s exact image duplicate crosses splits: %s and %s" %
                                 (name, first["image_path"], items[1]["image_path"]))
            ordered = sorted(
                items,
                key=lambda item: (-SPLIT_PRIORITY[item["split"]],
                                  item["row"]["image"], item["row_name"]))
            keeper = ordered[0]
            dropped = ordered[1:]
            for item in dropped:
                item["dropped"] = True
            report["duplicate_resolutions"].append({
                "sha256": image_hash,
                "labels_identical": True,
                "policy": "keep test > val > train; lexical path tie-break",
                "keeper": {
                    "split": keeper["split"],
                    "image": keeper["row"]["image"],
                    "source_path": keeper["image_path"],
                },
                "dropped": [{
                    "split": item["split"],
                    "image": item["row"]["image"],
                    "source_path": item["image_path"],
                } for item in dropped],
            })

        for items in dataset_hash_groups.values():
            for item in items:
                if item["dropped"]:
                    continue
                image_hash = item["image_hash"]
                global_previous = global_image_hashes.get(image_hash)
                if global_previous is not None and global_previous[0] != name:
                    global_duplicate_warnings.append({
                        "sha256": image_hash,
                        "first": global_previous[1],
                        "second": item["image_path"],
                    })
                else:
                    global_image_hashes[image_hash] = (name, item["image_path"])

    for split in SPLITS:
        source_split = os.path.join(source_root, split)
        annotation_path = os.path.join(source_split, "annotations.jsonl")
        kept = [item for item in prepared_by_split[split] if not item["dropped"]]
        canonical_rows = [item["row"] for item in kept]
        instance_histogram = collections.Counter(
            str(row["num_qrcodes"]) for row in canonical_rows)
        output_split = os.path.join(dataset_output, split)
        output_annotations = os.path.join(output_split, "annotations.jsonl")
        write_jsonl(output_annotations, canonical_rows)
        if sha256_file(annotation_path) != source_annotation_hashes[split]:
            raise RuntimeError("source annotations changed during conversion: %s" %
                               annotation_path)
        report["splits"][split] = {
            "images": len(canonical_rows),
            "source_rows": len(prepared_by_split[split]),
            "dropped_exact_duplicates": sum(
                item["dropped"] for item in prepared_by_split[split]),
            "instances": sum(row["num_qrcodes"] for row in canonical_rows),
            "negative_images": sum(row["num_qrcodes"] == 0 for row in canonical_rows),
            "instances_per_image": dict(sorted(instance_histogram.items())),
            "label_files_checked": sum(
                item["label_file_checked"] for item in kept),
            "source_annotations_sha256": source_annotation_hashes[split],
            "canonical_annotations_sha256": sha256_file(output_annotations),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True,
                        help="Repeat NAME=/absolute/source/root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label-tolerance", type=float, default=5e-4,
                        help="Normalized per-coordinate JSON/TXT tolerance")
    parser.add_argument("--skip-image-hash", action="store_true")
    parser.add_argument(
        "--deduplicate-identical-images", action="store_true",
        help=("For exact cross-split image duplicates, keep test then val then "
              "train only when canonical labels are exactly identical"))
    args = parser.parse_args()
    if args.skip_image_hash and args.deduplicate_identical_images:
        raise ValueError("--deduplicate-identical-images cannot be used with "
                         "--skip-image-hash")
    datasets = args.dataset
    names = [item[0] for item in datasets]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    output_root = os.path.abspath(args.output_root)
    if os.path.lexists(output_root):
        raise RuntimeError("Output already exists; refusing to overwrite: %s" % output_root)
    for name, source_root in datasets:
        if not os.path.isdir(source_root):
            raise ValueError("source dataset does not exist: %s=%s" % (name, source_root))
        if output_root == source_root or output_root.startswith(source_root + os.sep):
            raise ValueError("output-root must not be inside a source dataset: %s" %
                             source_root)
    staging = output_root + ".tmp-%d" % os.getpid()
    if os.path.lexists(staging):
        raise RuntimeError("staging path already exists: %s" % staging)
    os.makedirs(staging)
    try:
        reports = {}
        global_image_hashes = {}
        global_duplicate_warnings = []
        for name, source_root in datasets:
            reports[name] = convert_dataset(
                name, source_root, staging, args.label_tolerance,
                not args.skip_image_hash, args.deduplicate_identical_images,
                global_image_hashes,
                global_duplicate_warnings)
        report = {
            "schema_version": "qr_ordered_corners_v1",
            "output_root": output_root,
            "source_datasets_modified": False,
            "image_hashing_enabled": not args.skip_image_hash,
            "deduplicate_identical_images": args.deduplicate_identical_images,
            "cross_dataset_exact_duplicate_warnings": global_duplicate_warnings,
            "datasets": reports,
        }
        with open(os.path.join(staging, "conversion_report.json"), "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.rename(staging, output_root)
    except Exception:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
    print("PASS: canonical datasets written to %s" % output_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
