#!/usr/bin/env python3
"""Freeze and enforce final-only evaluation videos.

The two business videos are allowed only after the dataset, checkpoint and
inference thresholds have been frozen.  This guard records their hashes and
rejects dataset trees that contain video files, derived-looking filenames or
annotation metadata referring to either video.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import re


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-"))


def freeze_manifest(video_paths, output):
    if len(video_paths) < 1:
        raise ValueError("At least one --video is required")
    videos = []
    seen = set()
    for value in video_paths:
        path = os.path.realpath(os.path.abspath(value))
        if not os.path.isfile(path):
            raise IOError("Strict-eval video does not exist: %s" % path)
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError("Duplicate strict-eval video content: %s" % path)
        seen.add(digest)
        videos.append({
            "path": path,
            "basename": os.path.basename(path),
            "stem": normalized_stem(path),
            "size_bytes": os.path.getsize(path),
            "sha256": digest,
        })
    manifest = {
        "schema_version": "qr_strict_eval_v1",
        "policy": {
            "allowed": ["one final inference after checkpoint and thresholds are frozen"],
            "forbidden": [
                "training", "validation", "threshold selection", "calibration",
                "hard-negative mining", "background extraction", "augmentation",
                "model selection", "early stopping"
            ],
        },
        "videos": videos,
    }
    parent = os.path.dirname(os.path.abspath(output))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_manifest(path, verify_video_hashes=False):
    with open(path, "r") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "qr_strict_eval_v1":
        raise ValueError("Unsupported strict-eval manifest: %s" % path)
    if not manifest.get("videos"):
        raise ValueError("Strict-eval manifest has no videos")
    if verify_video_hashes:
        for item in manifest["videos"]:
            if not os.path.isfile(item["path"]):
                raise IOError("Strict-eval video moved or missing: %s" % item["path"])
            actual = sha256_file(item["path"])
            if actual != item["sha256"]:
                raise ValueError("Strict-eval video hash changed: %s" % item["path"])
    return manifest


def _forbidden_tokens(manifest):
    result = set()
    for item in manifest["videos"]:
        for value in (item.get("basename", ""), item.get("stem", "")):
            value = value.lower()
            if len(value) >= 4:
                result.add(value)
    return sorted(result)


def contains_forbidden_token(value, token):
    """Match a filename token without treating `fore` as `forest`."""
    pattern = r"(^|[^a-z0-9])%s([^a-z0-9]|$)" % re.escape(token.lower())
    return re.search(pattern, value.lower()) is not None


def check_dataset_roots(manifest_path, dataset_roots):
    manifest = load_manifest(manifest_path, verify_video_hashes=True)
    tokens = _forbidden_tokens(manifest)
    violations = []
    checked_files = 0
    for root_value in dataset_roots:
        root = os.path.realpath(os.path.abspath(root_value))
        if not os.path.isdir(root):
            violations.append("missing dataset root: %s" % root)
            continue
        for parent, _, names in os.walk(root):
            for name in names:
                checked_files += 1
                path = os.path.join(parent, name)
                relative = os.path.relpath(path, root).lower()
                if name.lower().endswith(VIDEO_EXTENSIONS):
                    violations.append("video file inside dataset: %s" % path)
                if any(contains_forbidden_token(relative, token)
                       for token in tokens):
                    violations.append("strict-eval name appears in dataset: %s" % path)
                if name == "annotations.jsonl":
                    with open(path, "r") as handle:
                        for line_number, line in enumerate(handle, 1):
                            lowered = line.lower()
                            if any(contains_forbidden_token(lowered, token)
                                   for token in tokens):
                                violations.append(
                                    "strict-eval reference in %s:%d" %
                                    (path, line_number))
    report = {
        "schema_version": "qr_strict_eval_check_v1",
        "manifest": os.path.realpath(os.path.abspath(manifest_path)),
        "dataset_roots": [os.path.realpath(os.path.abspath(x)) for x in dataset_roots],
        "checked_files": checked_files,
        "violations": violations,
        "pass": not violations,
    }
    if violations:
        raise ValueError("Strict-evaluation leakage check failed:\n- " +
                         "\n- ".join(violations[:50]))
    return report


def verify_final_input(manifest_path, video_path):
    manifest = load_manifest(manifest_path, verify_video_hashes=True)
    actual_path = os.path.realpath(os.path.abspath(video_path))
    actual_hash = sha256_file(actual_path)
    for item in manifest["videos"]:
        if actual_hash == item["sha256"]:
            return item
    raise ValueError("Input is not one of the frozen final-evaluation videos: %s" %
                     actual_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--video", action="append", required=True)
    freeze.add_argument("--output", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--manifest", required=True)
    check.add_argument("--dataset-root", action="append", required=True)
    final = subparsers.add_parser("verify-final-input")
    final.add_argument("--manifest", required=True)
    final.add_argument("--video", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_manifest(args.video, args.output)
    elif args.command == "check":
        result = check_dataset_roots(args.manifest, args.dataset_root)
    elif args.command == "verify-final-input":
        result = verify_final_input(args.manifest, args.video)
    else:
        parser.print_help()
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
