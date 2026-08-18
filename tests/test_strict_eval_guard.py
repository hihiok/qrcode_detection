#!/usr/bin/env python3
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strict_eval_guard import check_dataset_roots, freeze_manifest


def test_strict_eval_manifest_rejects_named_derived_frames():
    root = tempfile.mkdtemp(prefix="qr_guard_")
    try:
        video = os.path.join(root, "vrtest.mp4")
        with open(video, "wb") as handle:
            handle.write(b"not-a-real-video-but-hashable")
        manifest_path = os.path.join(root, "strict.json")
        freeze_manifest([video], manifest_path)
        dataset = os.path.join(root, "dataset")
        os.makedirs(dataset)
        with open(os.path.join(dataset, "safe.jpg"), "wb") as handle:
            handle.write(b"safe")
        with open(os.path.join(dataset, "forest.jpg"), "wb") as handle:
            handle.write(b"not-derived-from-fore")
        assert check_dataset_roots(manifest_path, [dataset])["pass"]
        with open(os.path.join(dataset, "vrtest_frame_0001.jpg"), "wb") as handle:
            handle.write(b"leak")
        try:
            check_dataset_roots(manifest_path, [dataset])
        except ValueError as exc:
            assert "leakage" in str(exc).lower()
        else:
            raise AssertionError("strict-eval derived name must fail")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_strict_eval_manifest_rejects_named_derived_frames()
    print("PASS")
