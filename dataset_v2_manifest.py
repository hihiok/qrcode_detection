#!/usr/bin/env python3
"""Build and verify a frozen, source-balanced QR V2 dataset manifest."""
from __future__ import print_function

import argparse
import hashlib
import json
import os

from qr_schema import read_jsonl, validate_canonical_row
from strict_eval_guard import check_dataset_roots


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_values(values, caster=str):
    result = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("Expected NAME=VALUE, got %s" % value)
        name, item = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError("Invalid or duplicate source name: %s" % name)
        result[name] = caster(item)
    return result


def inspect_split(root, split):
    split_root = os.path.join(root, split)
    annotation_path = os.path.join(split_root, "annotations.jsonl")
    if not os.path.isfile(annotation_path):
        raise IOError("Missing %s" % annotation_path)
    rows = read_jsonl(annotation_path)
    instances = negatives = 0
    image_paths = []
    image_digest = hashlib.sha256()
    for index, row in enumerate(rows, 1):
        image, _, _, values = validate_canonical_row(
            row, "%s:%d" % (annotation_path, index))
        path = os.path.join(split_root, image)
        if not os.path.isfile(path):
            raise IOError("Missing image referenced by %s: %s" %
                          (annotation_path, path))
        image_paths.append(os.path.realpath(path))
        image_digest.update(image.encode("utf-8"))
        image_digest.update(sha256_file(path).encode("ascii"))
        instances += len(values)
        negatives += int(len(values) == 0)
    return {
        "images": len(rows),
        "instances": instances,
        "negative_images": negatives,
        "annotations_sha256": sha256_file(annotation_path),
        "images_sha256": image_digest.hexdigest(),
        "annotation_path": os.path.realpath(annotation_path),
        "image_paths": image_paths,
    }


def build_manifest(source_values, weight_values, kind_values,
                   strict_eval_manifest, output):
    sources = parse_named_values(source_values, os.path.abspath)
    weights = parse_named_values(weight_values, float)
    kinds = parse_named_values(kind_values, str)
    if set(weights) != set(sources) or set(kinds) != set(sources):
        raise ValueError("--source, --weight and --kind must name identical sources")
    if any(value <= 0.0 for value in weights.values()):
        raise ValueError("All source weights must be positive")
    total_weight = sum(weights.values())
    roots = [os.path.realpath(path) for path in sources.values()]
    leakage_report = check_dataset_roots(strict_eval_manifest, roots)
    entries = []
    all_images = {}
    for name in sorted(sources):
        root = os.path.realpath(sources[name])
        approval = None
        if kinds[name] == "hard_negative":
            approval_path = os.path.join(root, "APPROVED_BY_HUMAN.txt")
            if not os.path.isfile(approval_path) or os.path.getsize(approval_path) == 0:
                raise ValueError(
                    "Hard-negative source requires manual preview approval: %s" %
                    approval_path)
            approval = {"path": os.path.realpath(approval_path),
                        "sha256": sha256_file(approval_path)}
        split_stats = {}
        for split in ("train", "val", "test"):
            stats = inspect_split(root, split)
            for image_path in stats.pop("image_paths"):
                if image_path in all_images:
                    raise ValueError(
                        "Image path leaks across sources/splits: %s (%s and %s/%s)" %
                        (image_path, all_images[image_path], name, split))
                all_images[image_path] = "%s/%s" % (name, split)
            split_stats[split] = stats
        entries.append({
            "name": name,
            "root": root,
            "kind": kinds[name],
            "sampling_weight": weights[name] / total_weight,
            "manual_approval": approval,
            "splits": split_stats,
        })
    manifest = {
        "schema_version": "qr_dataset_v2_manifest_v1",
        "strict_eval_manifest": os.path.realpath(strict_eval_manifest),
        "strict_eval_check": leakage_report,
        "sources": entries,
    }
    parent = os.path.dirname(os.path.abspath(output))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_and_verify_manifest(path):
    with open(path, "r") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "qr_dataset_v2_manifest_v1":
        raise ValueError("Unsupported dataset manifest: %s" % path)
    roots = [item["root"] for item in manifest.get("sources", [])]
    if not roots:
        raise ValueError("Dataset manifest has no sources")
    check_dataset_roots(manifest["strict_eval_manifest"], roots)
    for source in manifest["sources"]:
        approval = source.get("manual_approval")
        if approval:
            if (not os.path.isfile(approval["path"]) or
                    sha256_file(approval["path"]) != approval["sha256"]):
                raise ValueError("Hard-negative approval changed: %s" %
                                 source["name"])
        for split in ("train", "val", "test"):
            current = inspect_split(source["root"], split)
            expected = source["splits"][split]
            current.pop("image_paths")
            for key in ("images", "instances", "negative_images",
                        "annotations_sha256", "images_sha256"):
                if current[key] != expected[key]:
                    raise ValueError(
                        "Frozen dataset changed: %s/%s %s expected=%s actual=%s" %
                        (source["name"], split, key, expected[key], current[key]))
    return manifest


def training_sources(manifest_path):
    manifest = load_and_verify_manifest(manifest_path)
    return [(item["root"], float(item["sampling_weight"]), item["name"])
            for item in manifest["sources"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build")
    build.add_argument("--source", action="append", required=True,
                       help="NAME=/absolute/dataset/root")
    build.add_argument("--weight", action="append", required=True,
                       help="NAME=relative_sampling_weight")
    build.add_argument("--kind", action="append", required=True,
                       help="NAME=real|synthetic|negative|hard_negative")
    build.add_argument("--strict-eval-manifest", required=True)
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest(args.source, args.weight, args.kind,
                                args.strict_eval_manifest, args.output)
    elif args.command == "verify":
        result = load_and_verify_manifest(args.manifest)
    else:
        parser.print_help()
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
